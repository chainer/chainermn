import math

import chainer.cuda
import mpi4py.MPI

from chainermn.communicators import _base
from chainermn.communicators import _memory_utility
from chainermn import nccl


class NodeAwareCommunicator(_base.NodeAwareCommunicator):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        super(NodeAwareCommunicator, self).__init__(mpi_comm, use_nccl=True)
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def broadcast_data(self, model):
        self._init_comms()

        # TODO(akiba): use NCCL if necessary
        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize

        self.gpu_buffer_a.assign(n_bytes_total)
        _memory_utility.pack_params(
            params, itemsize, 'data', self.gpu_buffer_a)
        self.mpi_comm.Bcast(
            [self.gpu_buffer_a.buffer(n_bytes_total), mpi4py.MPI.FLOAT])
        _memory_utility.unpack_params(
            params, itemsize, 'data', self.gpu_buffer_a)

    def allreduce_grad(self, model, stream=chainer.cuda.Stream.null):
        self._init_comms()

        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_elems_per_node = int(math.ceil(n_elems_total / self.inter_size))
        n_bytes_per_node = n_elems_per_node * itemsize
        n_bytes_buffer = n_bytes_per_node * self.inter_size

        self.gpu_buffer_a.assign(n_bytes_buffer)
        self.gpu_buffer_b.assign(n_bytes_buffer)
        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        # Intra-node reduce
        self.intra_nccl_comm.reduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0, stream.ptr)

        # TODO(akiba): sync necessary?

        # Inter-node allreduce
        if self.intra_rank == 0:
            # Exchange all data to get own region data (bufferB -> bufferA)
            self.inter_mpi_comm.Alltoall(
                [self.gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
                [self.gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

            # Reduce own region data (inplace bufferA) and averaging
            ret = self.gpu_buffer_a.array(self.inter_size * n_elems_per_node) \
                      .reshape(self.inter_size, n_elems_per_node) \
                      .sum(axis=0) * (1.0 / self.size)

            # Gather others' region data (bufferA -> bufferB)
            for i in range(0, self.inter_size):
                self.gpu_buffer_a.from_device(
                    ret, n_bytes_per_node, i * n_bytes_per_node)
            self.inter_mpi_comm.Alltoall(
                [self.gpu_buffer_a.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT],
                [self.gpu_buffer_b.buffer(n_bytes_buffer), mpi4py.MPI.FLOAT])

        # Intra-node bcast
        self.intra_nccl_comm.bcast(
            self.gpu_buffer_b.ptr(), n_elems_total, nccl.NCCL_FLOAT, 0,
            stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
