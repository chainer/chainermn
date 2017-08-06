import nose.plugins.skip
import unittest

import chainer
import chainer.cuda
import chainer.links as L
import chainer.testing
import chainer.testing.attr
import chainermn
import numpy as np


class Cycle0SubA(chainer.Chain):
    def __init__(self, size):
        super(Cycle0SubA, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, x):
        return self.f(x)


class Cycle0SubB(chainer.Chain):
    def __init__(self, size):
        super(Cycle0SubB, self).__init__(
            f=L.Linear(size, 2))

    def __call__(self, h):
        return self.f(h)


class Cycle0(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle0, self).__init__(comm=comm)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)


class Cycle1Sub(chainer.Chain):
    def __init__(self, size):
        super(Cycle1Sub, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, h):
        return self.f(h)


class Cycle1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle1, self).__init__(comm=comm)
        self.add_link(Cycle1Sub(size), rank_in=rank_prev, rank_out=rank_next)


class Cross0(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cross0, self).__init__(comm=comm)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)


class Cross1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cross1, self).__init__(comm=comm)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)


class BranchSubA(chainer.Chain):
    def __init__(self, size):
        super(BranchSubA, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, x):
        return self.f(x)


class BranchSubB(chainer.Chain):
    def __init__(self, size):
        super(BranchSubB, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, *xs):
        x = xs[0]
        for _x in xs[1:]:
            x = x + _x
        return self.f(x)


class BranchParent1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent1, self).__init__(comm=comm)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_children)
        self.add_link(BranchSubB(size), rank_in=rank_children, rank_out=None)


class BranchParent2(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent2, self).__init__(comm=comm)
        ranks = [0] + rank_children
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=0, rank_out=0)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchParent3(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent3, self).__init__(comm=comm)
        ranks = rank_children + [0]
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=0, rank_out=0)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchParent4(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent4, self).__init__(comm=comm)
        ranks = rank_children + [0]
        ranks = ranks[1:] + ranks[0:1]
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=0, rank_out=0)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchChild(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_parent):
        super(BranchChild, self).__init__(comm=comm)
        self.add_link(
            BranchSubA(size),
            rank_in=rank_parent,
            rank_out=rank_parent)


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

    def test_cycle_model(self):
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

    def test_crossing_model(self):
        n, d = 100, 10
        X = np.random.randn(n, d).astype(np.float32)
        Y = (np.random.rand(n) * 2).astype(np.int32)

        if self.communicator.rank == 0:
            model = L.Classifier(Cross0(
                d, self.communicator, self.rank_next, self.rank_prev))
        else:
            model = L.Classifier(Cross1(
                d, self.communicator, self.rank_next, self.rank_prev))

        if self.gpu:
            model.to_gpu()
            X = chainer.cuda.to_gpu(X)
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()

    def check_branching_model(self, parent_model):
        n, d = 100, 10
        X = np.random.randn(n, d).astype(np.float32)
        Y = (np.random.rand(n) * 2).astype(np.int32)

        if self.communicator.rank == 0:
            rank_children = [rank for rank in range(1, self.communicator.size)]
            model = L.Classifier(parent_model(
                d, self.communicator, rank_children))
            if self.gpu:
                model.to_gpu()
                X = chainer.cuda.to_gpu(X)
                Y = chainer.cuda.to_gpu(Y)

            for i in range(n):
                err = model(X[i:i + 1], Y[i:i + 1])
                err.backward()
        else:
            model = BranchChild(d, self.communicator, 0)
            if self.gpu:
                model.to_gpu()

            for i in range(n):
                err = model()
                err.backward()

    def test_branching_model1(self):
        self.check_branching_model(BranchParent1)

    def test_branching_model2(self):
        self.check_branching_model(BranchParent2)

    def test_branching_model3(self):
        self.check_branching_model(BranchParent3)

    def test_branching_model4(self):
        self.check_branching_model(BranchParent4)
