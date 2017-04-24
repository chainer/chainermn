# Installation check for ChainerMN

This is a step-by-step guide to check if your environment is correctly set up.

## Single-node environment

### Basic MPI installation

Although ChainerMN stands for "Chainer MultiNode", it is good to start from single-node execution. 
First of all, you need MPI. If MPI is correctly installed, you will see `mpicc` and `mpiexec` command in your PATH.

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
    
If you see any error in above commands, please check your MPI installation.

### Check if MPI is CUDA-aware

Your MPI must be configured as `CUDA-aware`. You can use
`util/cuda_ware_check.c` to check your configuration.

```bash
$ make -C util
$ mpiexec -n 2 ./util/cuda_aware_check
OK.
```
    
### Check mpi4py

Next, you need to check if your MPI is configured to be
*CUDA-aware*. Both of Mvapich and Open MPI can be build to be
CUDA-aware.  (TODO: add check for CUDA-aware MPI)


Next, let's check mpi4py is correctly installed. You can use
`util/print_rank.py` for this purpose. The scripts prints ranks from
each MPI process in a sequential manner.

If you invoke the script without `mpiexec`, it just prints `0`.

```bash
$ python util/print_rank.py
0
```
    
If you run it with multiple processes, you will see the following output.

```bash
$ mpiexec -n 4 python util/print_rank.py
host00 0
host00 1
host00 2
host00 3
```

`host00` is the host name of the machine your are running the process.
If you get an output like below, it indicates something is wrong with
your installation.

```bash
# Wrong output !
$ mpiexec -n 4 python util/print_rank.py
host00 0
host00 0
host00 0
host00 0
```
    
A typical problem is that the `mpicc` used to build `mpi4py` and
`mpiexec` used to run the script are from different MPI installations.


## Multi-node environmnet

To use ChainerMN on multiple hosts, you need to login computing hosts
via ssh without password authentication (and preferreably without
username)

```bash
$ ssh host00 'hostname'
host00   # without hitting the password
```

Next step is to create a hostfile. A hostfile is a list of hosts on
which MPI processes run.

```bash
$ cat hostfile
host00
host01
host02
host03
```

Then, you can run your MPI program using the hostfile.

```bash
$ mpiexec -n 4 python util/print_rank.py
host00 0
host01 1
host02 2
host03 3
```

If you have multiple GPUs, you may want to run multiple processes on each host.
You can modify hostfile and specify number of processes to run on each host.

```bash
# If you are using Mvapich or Mpich:
$ cat hostfile
host00:4
host01:4
host02:4
host03:4

# If you are using Open MPI
$ cat hostfile
host00 cpu=4
host01 cpu=4
host02 cpu=4
host03 cpu=4
```

You can also specify computing hosts and resource mapping/binding using
command line options of `mpiexec`. Please refer to the MPI manual for more
advanced use of `mpiexec` command.



