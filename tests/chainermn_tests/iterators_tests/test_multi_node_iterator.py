import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
import numpy as np
import pytest
from six.moves import range
import unittest


class DummySerializer(chainer.serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(chainer.serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, np.ndarray):
            np.copyto(value, self.target[key])
        else:
            value = type(value)(np.asarray(self.target[key]))
        return value


@chainer.testing.parameterize(
    {'iterator_class': chainer.iterators.SerialIterator},
    {'iterator_class': chainer.iterators.MultiprocessIterator},
)
class TestMultiNodeIterator(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode only")

        self.N = 100
        self.dataset = np.arange(self.N).astype(np.float32)

    def test_mn_iterator(self):
        # Datasize is a multiple of batchsize.
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            self.iterator_class(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = self.communicator.mpi_comm.recv(
                            source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=0)

    def test_mn_iterator_frag(self):
        # Batasize is not a multiple of batchsize.
        bs = 7
        iterator = chainermn.iterators.create_multi_node_iterator(
            self.iterator_class(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == 0:
                    for rank_from in range(1, self.communicator.size):
                        _batch = self.communicator.mpi_comm.recv(
                            source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=0)

    def test_mn_iterator_change_master(self):
        # Check if it works under rank_master != 0.
        rank_master = 1
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            self.iterator_class(
                self.dataset, batch_size=bs, shuffle=True),
            self.communicator, rank_master)

        for e in range(3):
            for i in range(100):
                batch = iterator.next()
                if self.communicator.rank == rank_master:
                    rank_slaves = [i for i in range(self.communicator.size)
                                   if i != rank_master]
                    for rank_from in rank_slaves:
                        _batch = self.communicator.mpi_comm.recv(
                            source=rank_from)
                        self.assertEqual(batch, _batch)
                else:
                    self.communicator.mpi_comm.ssend(batch, dest=rank_master)

    def test_mn_iterator_no_repeat(self):
        # Do not repeat iterator to test if we can catch StopIteration.
        bs = 4
        iterator = chainermn.iterators.create_multi_node_iterator(
            self.iterator_class(
                self.dataset, batch_size=bs, shuffle=True, repeat=False),
            self.communicator)

        for e in range(3):
            try:
                while True:
                    batch = iterator.next()
                    if self.communicator.rank == 0:
                        for rank_from in range(1, self.communicator.size):
                            _batch = self.communicator.mpi_comm.recv(
                                source=rank_from)
                            self.assertEqual(batch, _batch)
                    else:
                        self.communicator.mpi_comm.ssend(batch, dest=0)
            except StopIteration:
                continue

    def test_overwrite_order(self):
        """Tests behavior on serialization.

        This test confirms that master's batch order can be overwritten,
        while slave's batch order cannot be overwritten, since slave must
        always distribute the completely same batch as master.
        """

        bs = 4
        rank_master = 0
        iterator = chainermn.iterators.create_multi_node_iterator(
            self.iterator_class(
                self.dataset, batch_size=bs, shuffle=True, repeat=False),
            self.communicator,
            rank_master=rank_master)

        order = iterator._order
        new_order = np.roll(order, 1)
        target = dict()
        iterator.serialize(DummySerializer(target))
        target['order'] = new_order
        iterator.serialize(DummyDeserializer(target))

        if self.communicator.rank == rank_master:
            self.assertEqual(iterator._order.tolist(), new_order.tolist())
        else:
            self.assertEqual(iterator._order.tolist(), order.tolist())
