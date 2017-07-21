import nose.plugins.skip
import unittest

import chainer
import chainer.cuda
import chainer.links as L
import chainer.testing
import chainer.testing.attr
import chainermn
import numpy as np


class Cycle0in(chainer.Chain):
    def __init__(self, size):
        super(Cycle0in, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, x):
        return self.f(x)


class Cycle0out(chainer.Chain):
    def __init__(self, size):
        super(Cycle0out, self).__init__(
            f=L.Linear(size, 2))

    def __call__(self, h):
        return self.f(h)


class Cycle0(chainermn.MultiNodeChainGroup):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle0, self).__init__(comm=comm)
        self.add_link(Cycle0in(size), rank_in=None, rank_out=rank_next)
        self.add_link(Cycle0out(size), rank_in=rank_prev, rank_out=None)


class Cycle1inst(chainer.Chain):
    def __init__(self, size):
        super(Cycle1inst, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, h):
        return self.f(h)


class Cycle1(chainermn.MultiNodeChainGroup):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle1, self).__init__(comm=comm)
        self.add_link(Cycle1inst(size), rank_in=rank_prev, rank_out=rank_next)


@chainer.testing.parameterize(
    {'gpu': True},
    {'gpu': False},
)
class TestMultiNodeChain(unittest.TestCase):

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

        self.rank_next = (self.communicator.rank + 1) % self.communicator.size
        self.rank_prev = (self.communicator.rank - 1) % self.communicator.size

    def test_cycle_forward(self):
        n, d = 100, 10

        if self.communicator.rank == 0:
            X = np.random.randn(n, d).astype(np.float32)
            Y = (np.random.rand(n) * 2).astype(np.int32)
            model = L.Classifier(
                Cycle0(d, self.communicator, self.rank_next, self.rank_prev))

            if self.gpu:
                model.to_gpu()
                X = chainer.cuda.to_gpu(X)
                Y = chainer.cuda.to_gpu(Y)

            for i in range(n):
                err = model(X[i:i + 1], Y[i:i + 1])
                err.backward()
        else:
            model = Cycle1(
                d, self.communicator, self.rank_next, self.rank_prev)
            if self.gpu:
                model.to_gpu()

            for i in range(n):
                err = model()
                err.backward()
