import unittest

import mpi4py.MPI
import numpy as np

from chainermn.communicators.naive_communicator import NaiveCommunicator

import copy
import nose.plugins.skip
import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import numpy

import chainermn
import chainermn.links


class ModelNormalBN(chainer.Chain):
    def __init__(self, n_in=3, n_units=3, n_out=2):
        super(ModelNormalBN, self).__init__(
            l1=chainer.links.Linear(n_in, n_units, nobias=True),
            bn1=chainer.links.BatchNormalization(n_units),
            l2=chainer.links.Linear(n_in, n_units, nobias=True),
            bn2=chainer.links.BatchNormalization(n_units),
            l3=chainer.links.Linear(n_in, n_out),
        )
        self.train = True

    def __call__(self, x):
        h = chainer.functions.relu(self.bn1(self.l1(x)))
        h = chainer.functions.relu(self.bn2(self.l2(h)))
        return self.l3(h)


class ModelDistributedBN(chainer.Chain):
    def __init__(self, comm, n_in=3, n_units=3, n_out=2):
        super(ModelDistributedBN, self).__init__(
            l1=chainer.links.Linear(n_in, n_units, nobias=True),
            bn1=chainermn.links.MultiNodeBatchNormalization(n_units, comm),
            l2=chainer.links.Linear(n_in, n_units, nobias=True),
            bn2=chainermn.links.MultiNodeBatchNormalization(n_units, comm),
            l3=chainer.links.Linear(n_in, n_out),
        )
        self.train = True

    def __call__(self, x):
        h = chainer.functions.relu(self.bn1(self.l1(x)))
        h = chainer.functions.relu(self.bn2(self.l2(h)))
        return self.l3(h)


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def test_multi_node_bn(self):
        comm = self.communicator

        local_batchsize = 10
        global_batchsize = 10 * comm.size
        ndim = 3
        np.random.seed(71)
        x = np.random.random((global_batchsize, ndim)).astype(np.float32)
        y = np.random.randint(0, 1, size=global_batchsize, dtype=np.int32)
        x_local = comm.mpi_comm.scatter(x.reshape(comm.size, local_batchsize, ndim))
        y_local = comm.mpi_comm.scatter(y.reshape(comm.size, local_batchsize))
        print(x.shape, y.shape, x_local.shape, y_local.shape)

        m1 = chainer.links.Classifier(ModelNormalBN())       # Single Normal
        m2 = chainer.links.Classifier(ModelNormalBN())       # Distributed Normal
        m3 = chainer.links.Classifier(ModelDistributedBN(comm))  # Distributed BN
        m4 = chainer.links.Classifier(ModelDistributedBN(comm))  # Sequential Normal
        m2.copyparams(m1)
        m3.copyparams(m1)
        m4.copyparams(m1)

        l1 = m1(x, y)
        m1.cleargrads()
        l1.backward()

        l2 = m2(x_local, y_local)
        m2.cleargrads()
        l2.backward()
        comm.allreduce_grad(m2)

        l3 = m3(x, y)
        m3.cleargrads()
        l3.backward()

        l4 = m4(x_local, y_local)
        m4.cleargrads()
        l4.backward()
        comm.allreduce_grad(m4)

        if comm.rank == 0:
            for p1, p2, p3, p4 in zip(
                    sorted(m1.namedparams()),
                    sorted(m2.namedparams()),
                    sorted(m3.namedparams()),
                    sorted(m4.namedparams())):
                name = p1[0]
                assert(p2[0] == name)
                assert(p3[0] == name)
                assert(p4[0] == name)

                # TODO: check p1[1].grad != p2[1].grad (to confirm that this test is valid)
                chainer.testing.assert_allclose(p1[1].grad, p3[1].grad)
                chainer.testing.assert_allclose(p1[1].grad, p4[1].grad)
