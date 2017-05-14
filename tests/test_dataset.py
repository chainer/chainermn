import mpi4py.MPI
import numpy as np
import unittest

import chainermn
from chainermn.communicators.naive_communicator import NaiveCommunicator


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def check_scatter_dataset(self, original_dataset):
        sub_dataset = chainermn.scatter_dataset(
            original_dataset, self.communicator)
        all_datasets = self.mpi_comm.gather(sub_dataset)

        if self.mpi_comm.rank == 0:
            # Test the total length
            total_size = sum(len(sub_dataset) for sub_dataset in all_datasets)
            self.assertEqual(len(original_dataset), total_size)

            # Test the length of each sub dataset
            expected_sub_dataset_size = len(
                original_dataset) // self.communicator.size
            for sub_dataset in all_datasets:
                self.assertGreaterEqual(
                    len(sub_dataset), expected_sub_dataset_size)
                self.assertLessEqual(
                    len(sub_dataset), expected_sub_dataset_size + 1)

            # Test the content of scattered datasets
            joined_dataset = sum((sub_dataset[:]
                                  for sub_dataset in all_datasets), [])
            self.assertEqual(joined_dataset, list(original_dataset[:]))

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
