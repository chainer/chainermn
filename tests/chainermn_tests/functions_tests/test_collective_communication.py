import chainer
import chainer.testing
import chainer.testing.attr
import numpy
import pytest
import unittest

import chainermn
import chainermn.functions


class TestCollectiveCommunication(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            self.device = self.communicator.intra_rank
            chainer.cuda.get_device(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            self.device = -1

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode")

    def check_all_to_all(self, xs):
        ys = chainermn.functions.all_to_all(self.communicator, xs, self.device)

        y = chainer.functions.sum(ys[0])
        for _y in ys[1:]:
            y += chainer.functions.sum(_y)

        y.backward()

        # Check if gradients are passed back without deadlock.
        self.assertTrue(xs[0].grad is not None)

    def test_all_to_all_cpu(self):
        self.setup(False)
        data = [
            chainer.Variable(numpy.zeros(
                (self.communicator.rank, i), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        self.check_all_to_all(data)

    @chainer.testing.attr.gpu
    def test_all_to_all_gpu(self):
        self.setup(True)

        chainer.cuda.get_device_from_id(self.device).use()
        data = [
            chainer.Variable(numpy.zeros(
                (self.communicator.rank, i), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        for x in data:
            x.to_gpu()
        self.check_all_to_all(data)

    def check_bcast(self, x):
        root = 0
        y = chainermn.functions.bcast(self.communicator, x, root, self.device)
        e = chainer.functions.mean_squared_error(y, x)
        e.backward()

        # Check backward does not fall in deadlock, and error = 0 in root.
        if self.communicator.rank == root:
            self.assertEqual(e.data, 0)
            self.assertEqual(e.grad, 1)

    def test_bcast_cpu(self):
        self.setup(False)
        x = chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
        self.check_bcast(x)

    @chainer.testing.attr.gpu
    def test_bcast_gpu(self):
        self.setup(True)
        x = chainer.Variable(
            numpy.random.normal(size=(100, 100)).astype(numpy.float32))
        x.to_gpu()
        self.check_bcast(x)
