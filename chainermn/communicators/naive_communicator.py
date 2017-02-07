import mpi4py.MPI

from chainermn.communicators import mpi_based_communicator


class NaiveCommunicator(mpi_based_communicator.MPIBasedCommunicator):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def broadcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Bcast(param.data)

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, param.grad)
            param.grad /= self.get_size()
