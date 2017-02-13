import chainer.datasets


class MPIBasedCommunicator(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm

    def broadcast_data(self, model):
        raise NotImplementedError()

    def allreduce_grad(self, model):
        raise NotImplementedError()

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    def scatter_dataset(self, dataset):
        # TODO(akiba): write why we do not use mpi_comm.scatter

        if self.rank == 0:
            mine = None
            n_samples = len(dataset)
            for i in range(self.size):
                b = n_samples * i // self.size
                e = n_samples * (i + 1) // self.size
                subds = chainer.datasets.SubDataset(dataset=dataset, start=b, finish=e)
                if i == 0:
                    mine = subds
                else:
                    self.mpi_comm.send(subds, dest=i)
            return mine
        else:
            return self.mpi_comm.recv(source=0)
