import chainer.cuda

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class PureNcclCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm):
        super(PureNcclCommunicator, self).__init__(mpi_comm, True)
        if nccl.get_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')
        self._init_ranks()

        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        self.intra_nccl_comm = None
        self.nccl_comm = None

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
            assert self.intra_nccl_comm is not None
            assert self.nccl_comm is not None
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=True)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        self.intra_nccl_comm = comms[2]
        self.nccl_comm = comms[3]

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model, stream=None):
        self._init_comms()
        if stream is None:
            stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params(model)
        itemsize = 4
        n_elems = sum(param.grad.size for param in params)
        n_bytes = itemsize * n_elems

        self.gpu_buffer_a.assign(n_bytes)
        self.gpu_buffer_b.assign(n_bytes)
        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)
        if stream != chainer.cuda.Stream.null:
            chainer.cuda.Stream.null.synchronize()
        self.nccl_comm.allReduce(self.gpu_buffer_a.ptr(),
                                 self.gpu_buffer_b.ptr(), n_elems,
                                 nccl.NCCL_FLOAT, nccl.NCCL_SUM,
                                 stream.ptr)
        if stream != chainer.cuda.Stream.null:
            stream.synchronize()
        ret = self.gpu_buffer_b.array(n_elems) * (1.0 / self.size)
        self.gpu_buffer_b.from_device(ret, n_bytes)
        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
