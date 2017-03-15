import chainer
import chainer.cuda
import chainer.links
import chainer.testing
import chainer.testing.attr
import mpi4py.MPI
import nose.plugins.skip
import numpy as np
import unittest

from chainermn.communicators.hierarchical_communicator \
    import HierarchicalCommunicator
from chainermn.communicators.naive_communicator import NaiveCommunicator
from chainermn.communicators.single_node_communicator \
    import SingleNodeCommunicator


class ExampleModel(chainer.Chain):

    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )


@chainer.testing.parameterize(
    {
        'communicator_class': NaiveCommunicator,
        'test_cpu': True,
        'test_gpu': True,
    }, {
        'communicator_class': HierarchicalCommunicator,
        'test_cpu': False,
        'test_gpu': True,
    }, {
        'communicator_class': SingleNodeCommunicator,
        'test_cpu': False,
        'test_gpu': True,
    }
)
class TestCommunicator(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = self.communicator_class(self.mpi_comm)

        if hasattr(self.communicator, 'intra_rank'):
            chainer.cuda.get_device(self.communicator.intra_rank).use()

    def test_rank(self):
        self.assertEqual(self.communicator.rank,
                         self.mpi_comm.Get_rank())

    def test_size(self):
        self.assertEqual(self.communicator.size,
                         self.mpi_comm.Get_size())

    def check_broadcast_data(self, model):
        model.a.W.data[:] = self.communicator.rank
        model.b.W.data[:] = self.communicator.rank + 1
        model.c.b.data[:] = self.communicator.rank + 2
        self.communicator.broadcast_data(model)
        chainer.testing.assert_allclose(model.a.W.data, 0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.data, 1 * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.data, 2 * np.ones((5, )))

    def check_allreduce_grad(self, model):
        model.a.W.grad[:] = self.communicator.rank
        model.b.W.grad[:] = self.communicator.rank + 1
        model.c.b.grad[:] = self.communicator.rank + 2
        self.communicator.allreduce_grad(model)

        base = (self.communicator.size - 1) / 2
        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * np.ones((5, )))

    def test_broadcast_data_cpu(self):
        if not self.test_cpu:
            raise nose.plugins.skip.SkipTest()
        model = ExampleModel()
        self.check_broadcast_data(model)

    @chainer.testing.attr.gpu
    def test_broadcast_data_gpu(self):
        if not self.test_gpu:
            raise nose.plugins.skip.SkipTest()
        model = ExampleModel()
        model.to_gpu()
        self.check_broadcast_data(model)

    def test_allreduce_grad_cpu(self):
        if not self.test_cpu:
            raise nose.plugins.skip.SkipTest()
        model = ExampleModel()
        self.check_allreduce_grad(model)

    @chainer.testing.attr.gpu
    def test_allreduce_grad_gpu(self):
        if not self.test_gpu:
            raise nose.plugins.skip.SkipTest()
        model = ExampleModel()
        model.to_gpu()
        self.check_allreduce_grad(model)
