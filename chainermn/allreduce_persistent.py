import chainer
import chainer.training.extension
import numpy as np


def _namedpersistents(model):
    assert isinstance(model, chainer.Link)

    for lname, link in model.namedlinks():
        for pname in link._persistent:
            yield lname + '/' + pname, link.__dict__[pname]


class AllreducePersistent(chainer.training.extension.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.extension.PRIORITY_WRITER + 1  # earlier than evaluator

    def __init__(self, model, comm):
        if hasattr(comm, 'mpi_comm'):
            comm = comm.mpi_comm

        self.model = model
        self.comm = comm

    def __call__(self, trainer=None):
        # We need to delay MPI4py import
        import mpi4py.MPI
        from chainermn.communicators._memory_utility import array_to_buffer_object

        for _, param in sorted(_namedpersistents(self.model)):
            if hasattr(param, 'dtype') and param.dtype == np.float32:
                buf = array_to_buffer_object(param)
                self.comm.Allreduce(mpi4py.MPI.IN_PLACE, buf)
                param /= self.comm.size
            else:
                pass  # TODO
