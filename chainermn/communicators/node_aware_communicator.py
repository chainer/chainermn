import cffi
import chainer.cuda
import collections
import ctypes
import cupy as cp
import math
import mpi4py.MPI
import numpy as np

from chainermn.communicators import naive_communicator
from chainermn import nccl


class _HostPinnedMemory(object):

    def __init__(self):
        self.size = 0
        self.memory = None
        self.cptr = None
        self.ffi = cffi.FFI()

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc_pinned_memory(size)
            self.cptr = self.ffi.cast('void *', self.memory.ptr)

    def ptr(self, offset=0):
        return ctypes.c_void_p(self.memory.ptr + offset)

    def buffer(self, size):
        return self.ffi.buffer(self.cptr, size)

    def array(self, count, offset=0):
        return np.frombuffer(
            self.memory, count=count, offset=offset, dtype=cp.float32)


class _DeviceMemory(object):

    def __init__(self):
        self.size = 0
        self.memory = None
        self.cptr = None
        self.ffi = cffi.FFI()

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc(size)

    def from_device(self, src, size, offset=0):
        dst = self.memory + offset
        dst.copy_from_device(src.data, size)

    def to_device(self, dst, size, offset=0):
        src = self.memory + offset
        dst.data.copy_from_device(src, size)

    def ptr(self):
        return self.memory.ptr

    def buffer(self, size):
        return self.ffi.buffer(self.ffi.cast('void *', self.memory.ptr), size)

    def array(self, shape, offset=0):
        return cp.ndarray(shape, memptr=self.memory + offset, dtype=cp.float32)


class NodeAwareCommunicator(naive_communicator.NaiveCommunicator):

    def __init__(
            self, mpi_comm=mpi4py.MPI.COMM_WORLD, use_cuda_aware_mpi=True):
        super(NodeAwareCommunicator, self).__init__(mpi_comm)
        self.use_cuda_aware_mpi = use_cuda_aware_mpi

        self._init_ranks()
        self._init_comms()

        self.cpu_buffer_a = _HostPinnedMemory()
        self.cpu_buffer_b = _HostPinnedMemory()
        self.gpu_buffer_a = _DeviceMemory()
        self.gpu_buffer_b = _DeviceMemory()

    def _init_ranks(self):
        global_names = self.mpi_comm.gather(mpi4py.MPI.Get_processor_name())

        if self.mpi_comm.rank == 0:
            name_to_global_ranks = collections.defaultdict(list)
            for global_rank, name in enumerate(global_names):
                name_to_global_ranks[name].append(global_rank)

            for global_ranks in name_to_global_ranks.values():
                global_ranks.sort()

            inter_names = sorted(
                set(global_names), key=lambda name: name_to_global_ranks[name])
            name_to_inter_rank = {
                name: inter_rank
                for inter_rank, name in enumerate(inter_names)
            }
            inter_size = len(inter_names)

            all_ranks = []
            for global_rank, name in enumerate(global_names):
                ranks = name_to_global_ranks[name]
                intra_rank = ranks.index(global_rank)
                intra_size = len(ranks)
                inter_rank = name_to_inter_rank[name]
                all_ranks.append((
                    global_rank, intra_rank, intra_size,
                    inter_rank, inter_size))
            my_ranks = self.mpi_comm.scatter(all_ranks)
        else:
            my_ranks = self.mpi_comm.scatter(None)

        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        self.intra_mpi_comm = self.mpi_comm.Split(
            self.inter_rank, self.intra_rank)

        if self.intra_rank == 0:
            inter_ranks = self.mpi_comm.allreduce([self.rank])
        else:
            inter_ranks = self.mpi_comm.allreduce([])

        world_group = self.mpi_comm.Get_group()
        inter_group = world_group.Incl(inter_ranks)
        self.inter_mpi_comm = self.mpi_comm.Create(inter_group)

        nccl_comm_id = self.intra_mpi_comm.bcast(nccl.NcclCommunicatorId())
        self.intra_nccl_comm = nccl.NcclCommunicator(
            self.intra_size, nccl_comm_id, self.intra_rank)

    def broadcast_data(self, model):
        # TODO(akiba): use NCCL if necessary
        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize

        self.gpu_buffer_a.assign(n_bytes_total)
        self._pack_params(params, itemsize, 'data', self.gpu_buffer_a)
        self.mpi_comm.Bcast(
            [self.gpu_buffer_a.buffer(n_bytes_total), mpi4py.MPI.FLOAT])
        self._unpack_params(params, itemsize, 'data', self.gpu_buffer_a)

    def allreduce_grad(self, model, stream=chainer.cuda.Stream.null):
        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_elems_per_node = int(math.ceil(n_elems_total / self.inter_size))
        n_bytes_per_node = n_elems_per_node * itemsize
        n_bytes_buffer = n_bytes_per_node * self.inter_size
        self._assign_buffers(n_bytes_buffer)

        self._pack_params(params, itemsize, 'grad', self.gpu_buffer_a)

        # Intra-node reduce
        self.intra_nccl_comm.reduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, 0, stream.ptr)

        # TODO(akiba): sync necessary?

        # Inter-node allreduce
        if self.intra_rank == 0:
            if self.use_cuda_aware_mpi:  # CUDA Aware
                self._allreduce_gradients_inter(
                    n_bytes_buffer, n_elems_per_node, n_bytes_per_node)
            else:
                raise NotImplementedError()

        # Intra-node bcast
        self.intra_nccl_comm.bcast(
            self.gpu_buffer_b.ptr(), n_elems_total, nccl.NCCL_FLOAT, 0,
            stream.ptr)

        self._unpack_params(params, itemsize, 'grad', self.gpu_buffer_b)

    def _assign_buffers(self, n_bytes_buffer):
        self.gpu_buffer_a.assign(n_bytes_buffer)
        self.gpu_buffer_b.assign(n_bytes_buffer)
        self.cpu_buffer_a.assign(n_bytes_buffer)
        self.cpu_buffer_b.assign(n_bytes_buffer)

    def _pack_params(self, params, itemsize, attr_name, buffer):
        offset = 0
        for param in params:
            grad = getattr(param, attr_name)
            size = grad.size * itemsize
            buffer.from_device(grad, size, offset)
            offset += size

    def _allreduce_gradients_inter(
            self, n_bytes_buffer, n_elems_per_node, n_bytes_per_node):
        # TODO(akiba): write why we use alltoall

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

    def _unpack_params(self, params, itemsize, attr_name, buffer):
        offset = 0
        for param in params:
            grad = getattr(param, attr_name)
            size = grad.size * itemsize
            buffer.to_device(grad, size, offset)
            offset += size
