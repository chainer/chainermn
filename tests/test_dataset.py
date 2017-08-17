import unittest

import mpi4py.MPI
import numpy as np

import chainermn
from chainermn.communicators.naive_communicator import NaiveCommunicator


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def check_scatter_dataset(self, original_dataset):
        my_dataset = chainermn.scatter_dataset(
            original_dataset, self.communicator)
        sub_datasets = self.mpi_comm.gather(my_dataset)

        if self.mpi_comm.rank == 0:
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

        self.check_scatter_dataset([])
        self.check_scatter_dataset([0])
        self.check_scatter_dataset(list(range(n)))
        self.check_scatter_dataset(list(range(n * 5 - 1)))

        self.check_scatter_dataset(np.array([]))
        self.check_scatter_dataset(np.array([0]))
        self.check_scatter_dataset(np.arange(n))
        self.check_scatter_dataset(np.arange(n * 5 - 1))
