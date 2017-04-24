# Installation check for ChainerMN

This is a step-by-step trouble shooting to check if your environment is correctly set up.

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

### Check SSH connection

To use ChainerMN on multiple hosts, you need to login computing hosts
via ssh without password authentication (and preferreably without
username), including the host you are currently logged in.

```bash
$ ssh host00 'hostname'
host00   # without hitting the password

$ ssh host01 'hostname'
host01   # without hitting the password

...
```

If your home directory is synched with NSF, you can achieve this just by doing:

```bash
$ cat ~/.ssh/id_rsa.pub >>~/.ssh/authorized_keys
```

Also, you need to pay attention to the environment variables on remote hosts.
MPI runtime connect to the remote hosts in *non-interactive* mode and environment
variables may differ from your interactive login session.

```bash
ssh host00 'env' | grep LD_LIBRARY_PATH
# Check the values and compare it to the local value.

ssh host01 'env' | grep LD_LIBRARY_PATH
# Check the values and compare it to the local value.

...
```

In particular, check the following variables, which are critical to
execute MPI programs:

    * `PATH`
    * `LD_LIBRARY_PATH`
    * `MV2_USE_CUDA`
    * `MV2_CPU_MAPPING`
    * `MV2_SMP_USE_CMA`
    
### Program files and data

When you run MPI programs, all hosts must have the program Python
binary and script files in the same path. First, check the python
binary and version are identical among hosts. Be careful if you are
using pyenv or Anaconda.

```bash
$ ssh host00 'which python; python --version'
/home/username/.pyenv/shims/python
Python 3.6.0 :: Anaconda 4.3.1 (64-bit)

$ ssh host01 'which python'
/home/username/.pyenv/shims/python
Python 3.6.0 :: Anaconda 4.3.1 (64-bit)

...
```

Also, the script file (and possibly data files) must be in the same
path on each host. 

```
$ ls yourscript.py
yourscript.py

$ ssh host00 "ls $PWD/yourscript.py"
/home/username/your/dir/yourscript.py

$ ssh host01 "ls $PWD/yourscript.py"
/home/username/your/dir/yourscript.py

...
```

If you are using NFS everything should be fine,
but if not you need to transfer all files manually.

### hostfile

Next step is to create a hostfile. A hostfile is a list of hosts on
which MPI processes run.

```bash
$ vi hostfile
$ cat hostfile
host00
host01
host02
host03
```

Then, you can run your MPI program using the hostfile.

```bash
$ mpiexec -n 4 --hostfile hostfile python util/print_rank.py
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

```bash
$ mpiexec -n 8 --hostfile hostfile python util/print_rank.py
host00 0
host00 1
host00 2
host00 3
host01 4
host01 5
host01 6
host01 7
```

You can also specify computing hosts and resource mapping/binding using
command line options of `mpiexec`. Please refer to the MPI manual for more
advanced use of `mpiexec` command.
