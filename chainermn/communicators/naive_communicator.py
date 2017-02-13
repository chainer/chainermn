import cffi
import chainer.cuda
import chainer.datasets
import mpi4py.MPI
import numpy as np


def _to_buffer(array):
    if chainer.cuda.get_array_module(array) is np:
        return array
    else:
        ffi = cffi.FFI()
        return (ffi.buffer(ffi.cast('void *', array.data.ptr), array.nbytes),
                mpi4py.MPI.FLOAT)


class NaiveCommunicator(object):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        self.mpi_comm = mpi_comm

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    def broadcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Bcast(_to_buffer(param.data))

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Allreduce(
                mpi4py.MPI.IN_PLACE, _to_buffer(param.grad))
            param.grad /= self.size

    def scatter_dataset(self, dataset):
        # TODO(akiba): write why we do not use mpi_comm.scatter

        if self.rank == 0:
            mine = None
            n_samples = len(dataset)
            for i in range(self.size):
                b = n_samples * i // self.size
                e = n_samples * (i + 1) // self.size
                subds = chainer.datasets.SubDataset(dataset, b, e)
                if i == 0:
                    mine = subds
                else:
                    self.mpi_comm.send(subds, dest=i)
            return mine
        else:
            return self.mpi_comm.recv(source=0)
