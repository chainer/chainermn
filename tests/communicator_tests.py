import chainer
import chainer.links
import chainer.testing
import chainer.testing.attr
import mpi4py.MPI
import numpy as np
import unittest

from chainermn import communicators


class ExampleModel(chainer.Chain):
    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )


@chainer.testing.parameterize(*chainer.testing.product({
    'communicator_class': [communicators.NaiveCommunicator]
}))
class TestCommunicator(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = self.communicator_class(self.mpi_comm)

    def test_rank(self):
        self.assertEqual(self.communicator.get_rank(),
                         self.mpi_comm.Get_rank())

    def test_size(self):
        self.assertEqual(self.communicator.get_size(),
                         self.mpi_comm.Get_size())

    def _test_broadcast_data(self, model):
        model.a.W.data[:] = self.communicator.get_rank()
        model.b.W.data[:] = self.communicator.get_rank() + 1
        model.c.b.data[:] = self.communicator.get_rank() + 2
        self.communicator.broadcast_data(model)
        chainer.testing.assert_allclose(model.a.W.data, 0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.data, 1 * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.data, 2 * np.ones((5, )))

    def _test_allreduce_grad(self, model):
        model.a.W.grad[:] = self.communicator.get_rank()
        model.b.W.grad[:] = self.communicator.get_rank() + 1
        model.c.b.grad[:] = self.communicator.get_rank() + 2
        self.communicator.allreduce_grad(model)

        base = (self.communicator.get_size() - 1) / 2
        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * np.ones((5, )))

    def test_broadcast_data_cpu(self):
        model = ExampleModel()
        self._test_broadcast_data(model)

    @chainer.testing.attr.gpu
    def test_broadcast_data_gpu(self):
        model = ExampleModel()
        model.to_gpu()
        self._test_broadcast_data(model)

    def test_allreduce_grad_cpu(self):
        model = ExampleModel()
        self._test_allreduce_grad(model)

    @chainer.testing.attr.gpu
    def test_allreduce_grad_gpu(self):
        model = ExampleModel()
        model.to_gpu()
        self._test_allreduce_grad(model)
