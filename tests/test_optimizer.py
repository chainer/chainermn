import unittest
import mock

import chainer
import chainermn
import chainer.testing
import chainer.testing.attr
import numpy as np


class ExampleModel(chainer.Chain):

    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )

class TestOptimizer(unittest.TestCase):

    def setUp(self):
        pass
    
    def setup_cpu(self):
        self.comm = chainermn.create_communicator('naive')
        self.target = ExampleModel()
        self.target.a.W.data[:] = self.comm.rank
        self.target.b.W.data[:] = self.comm.rank + 1
        self.target.c.W.data[:] = self.comm.rank + 2
        self.target.a.W.grad[:] = 0
        self.target.b.W.grad[:] = 0
        self.target.c.W.grad[:] = 0
        self.actual_optimizer = chainer.GradientMethod()
        self.actual_optimizer.create_update_rule = mock.MagicMock        
        self.actual_optimizer.setup(self.target)


    def setup_gpu(self, device=None):
        self.comm = chainermn.create_communicator('pure_nccl')
        device = self.comm.intra_rank
        chainer.cuda.get_device(device).use()
        self.target = ExampleModel()
        self.target.to_gpu()
        self.target.a.W.data[:] = self.comm.rank
        self.target.b.W.data[:] = self.comm.rank + 1
        self.target.c.W.data[:] = self.comm.rank + 2
        self.target.a.W.grad[:] = 0
        self.target.b.W.grad[:] = 0
        self.target.c.W.grad[:] = 0
        self.actual_optimizer = chainer.GradientMethod()
        self.actual_optimizer.create_update_rule = mock.MagicMock
        self.actual_optimizer.setup(self.target)


    def test_update_with_multi_node_optimizer_cpu(self):
        self.setup_cpu()
        self.optimizer = chainermn.create_multi_node_optimizer(self.actual_optimizer, self.comm)
        self.assertEqual(self.actual_optimizer.t, 0)
        self.optimizer.update()
        self.optimizer.target.a.W.grad[:] = self.comm.rank
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 1
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 2
        
        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 1)
        self.optimizer.target.a.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.a.W)
        self.optimizer.target.b.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.b.W)
        self.optimizer.target.c.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.c.W)

        base = (self.comm.size - 1.0) / 2
        chainer.testing.assert_allclose(self.optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

    
    @chainer.testing.attr.gpu
    def test_update_with_multi_node_optimizer_gpu(self):
        self.setup_gpu()
        self.optimizer = chainermn.create_multi_node_optimizer(self.actual_optimizer, self.comm)
        self.assertEqual(self.actual_optimizer.t, 0)
        self.optimizer.update()
        self.optimizer.target.a.W.grad[:] = self.comm.rank
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 1
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 2
        
        self.optimizer.update()
        self.assertEqual(self.actual_optimizer.t, 1)
        self.optimizer.target.a.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.a.W)
        self.optimizer.target.b.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.b.W)
        self.optimizer.target.c.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.c.W)

        base = (self.comm.size - 1.0) / 2
        chainer.testing.assert_allclose(self.optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

    @chainer.testing.attr.gpu
    def test_update_with_double_buffering_optimizer_gpu(self):
        self.setup_gpu()
        self.optimizer = chainermn.create_multi_node_optimizer(self.actual_optimizer, self.comm, double_buffering=True)
        self.assertEqual(self.actual_optimizer.t, 0)
        self.optimizer.update()
        self.optimizer.target.a.W.grad[:] = self.comm.rank
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 1
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 2
        
        self.optimizer.update()
        self.optimizer.wait()
        self.assertEqual(self.actual_optimizer.t, 0)
        base = (self.comm.size - 1.0) / 2
        chainer.testing.assert_allclose(self.optimizer.communicated_target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.communicated_target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.communicated_target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

        self.optimizer.target.a.W.grad[:] = self.comm.rank + 3
        self.optimizer.target.b.W.grad[:] = self.comm.rank + 4
        self.optimizer.target.c.W.grad[:] = self.comm.rank + 5
        self.optimizer.update()
        self.optimizer.wait()
        self.assertEqual(self.actual_optimizer.t, 1)
        self.optimizer.target.a.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.a.W)
        self.optimizer.target.b.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.b.W)
        self.optimizer.target.c.W.update_rule.update.assert_called_once_with(
            self.optimizer.target.c.W)
        chainer.testing.assert_allclose(self.optimizer.communicated_target.a.W.grad,
                                        (base + 3) * np.ones((3, 2)))
        chainer.testing.assert_allclose(self.optimizer.communicated_target.b.W.grad,
                                        (base + 4) * np.ones((4, 3)))
        chainer.testing.assert_allclose(self.optimizer.communicated_target.c.W.grad,
                                        (base + 5) * np.ones((5, 4)))

      
