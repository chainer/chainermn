import cffi
import chainer.cuda
import mpi4py.MPI
import numpy as np

from chainermn.communicators import mpi_based_communicator


def _to_buffer(array):
    if chainer.cuda.get_array_module(array) is np:
        return array
    else:
        ffi = cffi.FFI()
        return (ffi.buffer(ffi.cast('void *', array.data.ptr), array.nbytes),
                mpi4py.MPI.FLOAT)


class NaiveCommunicator(mpi_based_communicator.MPIBasedCommunicator):

    def __init__(self, mpi_comm=mpi4py.MPI.COMM_WORLD):
        super(NaiveCommunicator, self).__init__(mpi_comm)

    def broadcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Bcast(_to_buffer(param.data))

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            self.mpi_comm.Allreduce(
                mpi4py.MPI.IN_PLACE, _to_buffer(param.grad))
            param.grad /= self.get_size()
