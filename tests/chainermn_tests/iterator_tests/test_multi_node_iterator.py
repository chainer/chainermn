import chainer
import chainer.testing
import chainermn
import numpy as np
import pytest
import unittest


class TestMultiNodeIterator(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode only")

        self.N = 100
        self.dataset = np.arange(self.N).astype(np.float32)

    def test_mn_iterator(self):
        # Batasize is a multiple of batchsize.
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = chainermn.functions.recv(
                            self.communicator, rank=rank_from)
                        chainer.testing.assert_allclose(
                            batch.data, _batch.data)
                else:
                    chainermn.functions.send(batch, self.communicator, rank=0)

    def test_mn_iterator_frag(self):
        # Batasize is not a multiple of batchsize.
        bs = 7
        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = chainermn.functions.recv(
                            self.communicator, rank=rank_from)
                        chainer.testing.assert_allclose(
                            batch.data, _batch.data)
                else:
                    chainermn.functions.send(batch, self.communicator, rank=0)

    def test_mn_iterator_change_master(self):
        # Check if it works under rank_master != 0.
        rank_master = 1
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator, rank_master)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == rank_master:
                    rank_slaves = [i for i in range(self.communicator.size)
                                   if i != rank_master]
                    for rank_from in rank_slaves:
                        _batch = chainermn.functions.recv(
                            self.communicator, rank=rank_from)
                        chainer.testing.assert_allclose(
                            batch.data, _batch.data)
                else:
                    chainermn.functions.send(
                        batch, self.communicator, rank=rank_master)

    def test_mn_iterator_no_repeat(self):
        # Do not repeat iterator to test if we can catch StopIteration.
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                self.dataset, batch_size=bs, shuffle=True, repeat=False),
            self.communicator)

        for e in range(3):
            try:
                while True:
                    batch = iterator.next()
                    if self.communicator.rank == 0:
                        for rank_from in range(1, self.communicator.size):
                            _batch = chainermn.functions.recv(
                                self.communicator, rank=rank_from)
                            chainer.testing.assert_allclose(
                                batch.data, _batch.data)
                    else:
                        chainermn.functions.send(
                            batch, self.communicator, rank=0)
            except StopIteration:
                continue
