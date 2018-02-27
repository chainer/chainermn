import itertools
import numpy
import pytest
import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import chainermn


def debug(*s):
    import sys
    from mpi4py.MPI import COMM_WORLD
    print('[rank:{}]'.format(COMM_WORLD.Get_rank()), *s, file=sys.stderr, flush=True)


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
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


class TestIteratorCompatibility(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')
        self.device = -1

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode only")

        self.N = 6
        self.dataset = [numpy.array(i, dtype=numpy.float32)
                        for i in range(self.N)]
        self.bs = 2

    def test_iterator_compatibility(self):
        iters = (
            lambda: chainer.iterators.SerialIterator(
                        self.dataset, batch_size=self.bs, shuffle=True),
            lambda: chainermn.iterators.create_multi_node_iterator(
                        chainer.iterators.SerialIterator(
                            self.dataset, batch_size=self.bs, shuffle=True),
                        self.communicator, device=self.device),
        )

        bs_n_ratio = self.bs / self.N

        for it_before, it_after in itertools.permutations(iters, 2):
            it = it_before()

            self.assertEqual(it.epoch, 0)
            self.assertAlmostEqual(it.epoch_detail, 0 * bs_n_ratio)
            batch1 = it.next()
            self.assertEqual(len(batch1), self.bs)
#            self.assertIsInstance(batch1, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 1 * bs_n_ratio)
            batch2 = it.next()
            self.assertEqual(len(batch2), self.bs)
#            self.assertIsInstance(batch2, list)
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

            target = dict()
            it.serialize(DummySerializer(target))

            it = it_after()
            it.serialize(DummyDeserializer(target))
            self.assertFalse(it.is_new_epoch)
            self.assertAlmostEqual(it.epoch_detail, 2 * bs_n_ratio)

            batch3 = it.next()
            self.assertEqual(len(batch3), self.bs)
#            self.assertIsInstance(batch3, list)
            self.assertTrue(it.is_new_epoch)
#            self.assertEqual(sorted(batch1 + batch2 + batch3), self.dataset)
            self.assertAlmostEqual(it.epoch_detail, 3 * bs_n_ratio)
        debug('end')
