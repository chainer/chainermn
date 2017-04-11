import mpi4py.MPI

from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility


class NaiveCommunicator(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            buf = _memory_utility.array_to_buffer_object(param.grad)
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
            param.grad /= self.size
