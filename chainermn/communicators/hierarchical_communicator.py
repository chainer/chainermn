import chainer.cuda
import math

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class HierarchicalCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm):
        super(HierarchicalCommunicator, self).__init__(mpi_comm, use_nccl=True)
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params(model)
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

        # Inter-node allreduce
        if self.intra_rank == 0:
            _communication_utility.inter_allreduce_gpu(
                self.inter_mpi_comm, self.size,
                self.gpu_buffer_a, self.gpu_buffer_b,
                n_bytes_buffer, n_elems_per_node, n_bytes_per_node, stream)

        # Intra-node bcast
        self.intra_nccl_comm.bcast(
            self.gpu_buffer_b.ptr(), n_elems_total, nccl.NCCL_FLOAT, 0,
            stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
