import mpi4py.MPI
import numpy

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class CommunicatorBase(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    def send(self, array, dest, tag):
        shape = numpy.array(array.shape, dtype='int')
        buf = _memory_utility.array_to_buffer_object(array.astype('float32'))
        self.mpi_comm.Send([shape, mpi4py.MPI.INT], dest=dest, tag=tag)
        self.mpi_comm.Send(buf, dest=dest, tag=tag)

    def recv(self, source, tag):
        shape = numpy.empty(2, dtype='int')
        self.mpi_comm.Recv([shape, mpi4py.MPI.INT], source=source, tag=tag)
        buf = numpy.empty(shape[0] * shape[1], dtype='float32')
        self.mpi_comm.Recv(buf, source=source, tag=tag)
        return buf.reshape(shape)

    def broadcast_data(self, model):
        raise NotImplementedError()

    def allreduce_grad(self, model):
        raise NotImplementedError()


class NodeAwareCommunicatorBase(CommunicatorBase):

    def __init__(self, mpi_comm, use_nccl):
        super(NodeAwareCommunicatorBase, self).__init__(mpi_comm)

        if use_nccl and not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL can be found by dynamic linkers, '
                'and ChainerMN is installed without --no-nccl flag.'
            )

        self.use_nccl = use_nccl

        self._init_ranks()

        # TODO(akiba): write why we delay initializing comms
        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        if self.use_nccl:
            self.intra_nccl_comm = None

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
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=self.use_nccl)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        if self.use_nccl:
            self.intra_nccl_comm = comms[2]
