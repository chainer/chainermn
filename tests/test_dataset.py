import unittest

import mpi4py.MPI
import nose.plugins.skip
import numpy as np
from nose.plugins.attrib import attr

import chainermn
from chainermn.communicators.naive_communicator import NaiveCommunicator
from chainermn.datasets import DataSizeError
from chainermn.datasets import scatter_dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def check_scatter_dataset(self, original_dataset, shuffle=False, root=0):
        my_dataset = chainermn.scatter_dataset(
            original_dataset, self.communicator,
            shuffle=shuffle, root=root)
        sub_datasets = self.mpi_comm.gather(my_dataset, root=root)

        if self.mpi_comm.rank == root:
            # Test the sizes
            sub_sizes = [len(sub_dataset) for sub_dataset in sub_datasets]
            self.assertEqual(len(set(sub_sizes)), 1)
            sub_size = sub_sizes[0]
            self.assertLessEqual(
                len(original_dataset), sub_size * self.mpi_comm.size)
            self.assertGreater(
                len(original_dataset), (sub_size - 1) * self.mpi_comm.size)

            # Test the content of scattered datasets
            joined_dataset = sum((sub_dataset[:]
                                  for sub_dataset in sub_datasets), [])
            self.assertEqual(set(joined_dataset), set(original_dataset))

    def test_scatter_dataset(self):
        n = self.communicator.size

        for shuffle in [True, False]:
            for root in range(self.communicator.size):
                self.check_scatter_dataset([], shuffle, root)
                self.check_scatter_dataset([0], shuffle, root)
                self.check_scatter_dataset(list(range(n)), shuffle, root)
                self.check_scatter_dataset(list(range(n * 5 - 1)),
                                           shuffle, root)

                self.check_scatter_dataset(np.array([]), shuffle, root)
                self.check_scatter_dataset(np.array([0]), shuffle, root)
                self.check_scatter_dataset(np.arange(n), shuffle, root)
                self.check_scatter_dataset(np.arange(n * 5 - 1), shuffle, root)

    def scatter_large_data(self, comm_type):
        comm = self.communicator
        if comm.rank == 0:
            data = ["test"] * 2000000000
            data = chainermn.scatter_dataset(data, comm)
        else:
            data = []
            data = scatter_dataset(data, comm)

    @attr(slow=True)
    def test_scatter_large_dataset(self):
        # This test only runs when comm.size >= 2.
        if self.communicator.size == 1:
            raise nose.plugins.skip.SkipTest()

        # This test inherently requires large memory (>4GB) and
        # we skip this test so far.
        for comm_type in ['naive', 'flat']:
            self.assertRaises(DataSizeError,
                              lambda: self.scatter_large_data(comm_type))
