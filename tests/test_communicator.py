import unittest

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
from chainermn.communicators.nccl_communicator \
    import NcclCommunicator
from chainermn.communicators.non_cuda_aware_communicator \
    import NonCudaAwareCommunicator
from chainermn.communicators.single_node_communicator \
    import SingleNodeCommunicator
from chainermn.communicators.two_dimensional_communicator \
    import TwoDimensionalCommunicator
from chainermn import nccl


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
        'gpu': True,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
    }, {
        'communicator_class': HierarchicalCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl': True,
    }, {
        'communicator_class': TwoDimensionalCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl': True,
    }, {
        'communicator_class': SingleNodeCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': False,
        'nccl': True,
    }, {
        'communicator_class': NonCudaAwareCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl': True,
    }, {
        'communicator_class': NcclCommunicator,
        'test_cpu': False,
        'test_gpu': True,
        'multi_node': True,
        'nccl': True,
        'nccl1': False,
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
        if hasattr(self, 'nccl1') and not self.nccl1 \
           and nccl.get_version() < 2000:
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

    def check_send_and_recv(self, *shape):
        if self.communicator.size < 2:
            raise nose.plugins.skip.SkipTest()

        if self.communicator.rank > 0:
            rank_prev = (self.communicator.rank - 1) % self.communicator.size
            data_recv = self.communicator.recv(source=rank_prev, tag=0)
            chainer.testing.assert_allclose(
                data_recv, rank_prev * np.ones((shape)))

        if self.communicator.rank < self.communicator.size - 1:
            rank_next = (self.communicator.rank + 1) % self.communicator.size
            data_send = self.communicator.rank * \
                np.ones((shape)).astype(np.float32)
            self.communicator.send(data_send, dest=rank_next, tag=0)

    def test_send_and_recv1(self):
        self.check_send_and_recv(50)

    def test_send_and_recv2(self):
        self.check_send_and_recv(50, 20)

    def test_send_and_recv3(self):
        self.check_send_and_recv(50, 20, 5)

    def test_send_and_recv4(self):
        self.check_send_and_recv(50, 20, 5, 3)

    def check_broadcast_data(self, model):
        model.a.W.data[:] = self.communicator.rank
        model.b.W.data[:] = self.communicator.rank + 1
        model.c.b.data[:] = self.communicator.rank + 2
        self.communicator.broadcast_data(model)
        chainer.testing.assert_allclose(model.a.W.data, 0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.data, 1 * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.data, 2 * np.ones((5, )))

    def check_allreduce_grad(self, model):
        # We need to repeat twice for regressions on lazy initialization of
        # sub communicators.
        for _ in range(2):
            model.a.W.grad[:] = self.communicator.rank
            model.b.W.grad[:] = self.communicator.rank + 1
            model.c.b.grad[:] = self.communicator.rank + 2

            self.communicator.allreduce_grad(model)
            base = (self.communicator.size - 1.0) / 2

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
