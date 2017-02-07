class MPIBasedCommunicator(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm

    def broadcast_data(self, model):
        raise NotImplementedError()

    def allreduce_grad(self, model):
        raise NotImplementedError()

    def get_rank(self):
        return self.mpi_comm.Get_rank()

    def get_size(self):
        return self.mpi_comm.Get_size()
