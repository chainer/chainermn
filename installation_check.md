# Installation check for ChainerMN

This is a step-by-step guide to check if your environment is correctly set up.

## Single-node environment

Although ChainerMN stands for "Chainer MultiNode", it is good to start from single-node execution. 
First of all, you need MPI. If MPI is correctly installed, you will see `mpicc` and `mpiexec`command.

Below is an example of output from Mvapich on Linux.

    $ which mpicc
    /usr/local/bin/mpicc

    $ mpicc -show
    gcc -I/usr/local/include ...(snip)... -lmpi

    $ which mpiexec
    /usr/local/bin/mpiexec
    
    $ mpiexec --version
    HYDRA build details:
    Version:                                 3.1.4
    Release Date:                            Wed Sep  7 14:33:43 EDT 2016
    CC:                              gcc
    CXX:                             g++
    F77:
    F90:
    Configure options:  (snip)
    Process Manager:                         pmi
    Launchers available:                     ssh rsh fork slurm ll lsf sge manual persist
    Topology libraries available:            hwloc
    Resource management kernels available:   user slurm ll lsf sge pbs cobalt
    Checkpointing libraries available:
    Demux engines available:                 poll select
    

## Multi-node environmnet

## Check if MPI is working

First, you need an MPI runtime to use ChainerMN. 
