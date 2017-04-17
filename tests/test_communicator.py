import unittest
import sys

import mpi4py.MPI
import nose.plugins.skip
import numpy as np

import chainer
import chainer.cuda
import chainer.links
import chainer.testing
import chainer.testing.attr
from chainermn.communicators import _communication_utility
from chainermn.communicators.flat_communicator \
    import FlatCommunicator
from chainermn.communicators.hierarchical_communicator \
    import HierarchicalCommunicator
from chainermn.communicators.naive_communicator \
    import NaiveCommunicator
from chainermn.communicators.single_node_communicator \
    import SingleNodeCommunicator
from chainermn.communicators.two_dimensional_communicator \
    import TwoDimensionalCommunicator


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
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'gpu' : True,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
    }, {
        'communicator_class': HierarchicalCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl' : True,
    }, {
        'communicator_class': TwoDimensionalCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl' : True,
    }, {
        'communicator_class': SingleNodeCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': False,
        'nccl' : True,
    }
)
class TestCommunicator(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD

        if not self.multi_node:
            ranks = _communication_utility.init_ranks(self.mpi_comm)
            inter_size = ranks[4]
            if inter_size > 1:
                raise nose.plugins.skip.SkipTest()

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

        sys.stdout.flush()
        for i in range(self.communicator.size):
            if self.communicator.rank == i:
                sys.stderr.write("\n--- Rank: {}/{}\n".format(i, self.communicator.size))
                sys.stderr.write("a.W.grad = {}\n".format(model.a.W.grad))
                sys.stderr.write("b.W.grad = {}\n".format(model.b.W.grad))
                sys.stderr.write("c.W.grad = {}\n".format(model.c.b.grad))
                sys.stderr.write("base = {}\n".format(base))
                sys.stderr.write("(base+0) * np.ones((3,2)) = {}\n".format((base + 0) * np.ones((3, 2))))
            self.communicator.mpi_comm.Barrier()

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
