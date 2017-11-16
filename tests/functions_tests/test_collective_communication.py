import copy
import nose.plugins.skip
import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import numpy

import chainermn
import chainermn.functions


@chainer.testing.parameterize(
    {'gpu': True},
    {'gpu': False},
)
class TestPointToPointCommunication(unittest.TestCase):

    def setUp(self):
        if self.gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            device = self.communicator.intra_rank
            chainer.cuda.get_device(device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            device = -1

        if self.communicator.size < 2:
            raise nose.plugins.skip.SkipTest()

        self.device = device

    def check_all_to_all(self, xs):
        ys = chainermn.functions.all_to_all(self.communicator, xs, self.device)

        y = chainer.functions.sum(ys[0])
        for _y in ys[1:]:
            y += chainer.functions.sum(_y)

        y.backward()

        self.assertIsNotNone(xs[0].grad)

    def test_all_to_all_cpu(self):
        data = [
            chainer.Variable(numpy.zeros((self.communicator.rank, i), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        self.check_all_to_all(data)

    @chainer.testing.attr.gpu
    def test_all_to_all_gpu(self):
        if not self.gpu:
            raise nose.plugins.skip.SkipTest()
        chainer.cuda.get_device_from_id(self.device).use()
        data = [
            chainer.Variable(numpy.zeros((self.communicator.rank, i), dtype=numpy.float32))
            for i in range(self.communicator.size)]
        for x in data:
            x.to_gpu()
        self.check_all_to_all(data)
