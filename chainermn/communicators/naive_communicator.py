import mpi4py.MPI

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility


class NaiveCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model):
        for param in _memory_utility.extract_params(model):
            buf = _memory_utility.array_to_buffer_object(param.grad)
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
            param.grad /= self.size
