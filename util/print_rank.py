# coding: utf-8

import os

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

for i in range(size):
    if i == rank:
        print("{} {}".format(os.uname()[1], i))
    comm.Barrier()

