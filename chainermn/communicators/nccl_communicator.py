import chainer.cuda
import math

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class NcclCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm):
        super(NcclCommunicator, self).__init__(mpi_comm)
        self.use_nccl = True

        self._init_ranks()

        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        if self.use_nccl:
            self.inter_nccl_comm = None

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        if self.inter_mpi_comm is not None:
            assert self.intra_mpi_comm is not None
            assert not self.use_nccl or self.inter_nccl_comm is not None
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=self.use_nccl)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        if self.use_nccl:
            self.inter_nccl_comm = comms[3]

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

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
        self.inter_nccl_comm.allreduce(self.gpu_buffer_a.ptr(),
                                       self.gpu_buffer_b.ptr(), n_elems_total,
                                       nccl.NCCL_FLOAT, nccl.NCCL_SUM,
                                       stream.ptr)
        stream.synchronize()
        ret = self.gpu_buffer_b.array(n_elems_total) * (1.0 / self.size)
        self.gpu_buffer_b.from_device(ret, n_bytes_buffer)
        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
