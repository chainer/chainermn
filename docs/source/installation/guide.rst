Installation Guide
==================

Requirements
------------
In addition to Chainer, ChainerMN depends on the following software libraries:
CUDA-Aware MPI, NVIDIA NCCL, and a few Python packages including MPI4py.


Chainer
~~~~~~~

ChainerMN adds distributed training features to Chainer;
thus it naturally requires Chainer.
Please refer to `the official instructions <http://docs.chainer.org/en/latest/install.html>`_ to install.



.. _mpi-install:

CUDA-Aware MPI
~~~~~~~~~~~~~~

ChainerMN relies on MPI.
In particular, for efficient communication between GPUs, it uses CUDA-aware MPI.
For details about CUDA-aware MPI, see `this introduction article <https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/>`_.
(If you use only the CPU mode, MPI does not need to be CUDA-Aware. See :ref:`non-gpu-env` for more details.)

The CUDA-aware features depend on several MPI packages, which need to be configured and built properly.
The following are examples of Open MPI and MVAPICH.

Open MPI (for details, see `the official instructions <https://www.open-mpi.org/faq/?category=building#build-cuda>`_)::

  $ ./configure --with-cuda
  $ make -j4
  $ sudo make install

MVAPICH (for details, see `the official instructions <http://mvapich.cse.ohio-state.edu/static/media/mvapich/mvapich2-2.0-userguide.html#x1-120004.5>`_)::

  $ ./configure --enable-cuda
  $ make -j4
  $ sudo make install
  $ export MV2_USE_CUDA=1  # Should be set all the time when using ChainerMN

.. _nccl-install:
  
NVIDIA NCCL
~~~~~~~~~~~

To enable efficient intra-node GPU-to-GPU communication,
we use `NVIDIA NCCL <https://github.com/NVIDIA/nccl>`_.
See `the official instructions <https://github.com/NVIDIA/nccl#build--run>`_ for installation.

Please properly configure environment variables to expose NCCL both when you install and use ChainerMN.
Typical configurations should look like the following::

  export NCCL_ROOT=<path to NCCL directory>
  export CPATH=$NCCL_ROOT/include:$CPATH
  export LD_LIBRARY_PATH=$NCCL_ROOT/lib/:$LD_LIBRARY_PATH
  export LIBRARY_PATH=$NCCL_ROOT/lib/:$LIBRARY_PATH

ChainerMN requires NCCL even if you have only one GPU per node.  The
only exception is when you run ChainerMN on CPU-only environments. See
:ref:`non-gpu-env` for more details.

.. _mpi4py-install:

MPI4py
~~~~~~

ChainerMN depends on a few Python packages, which are listed in ``requirements.txt``.
They are automatically installed when you install ChainerMN via PyPI.

However, among them, we need to be a little careful about MPI4py.
It links to MPI at installation time, so please be sure
to properly configure environment variables
so that MPI is available at installation time.
In particular, if you have multiple MPI implementations in your environment,
please expose the implementation that you want to use
both when you install and use ChainerMN.


.. _chainermn-install:

Installation
------------

Install via pip
~~~~~~~~~~~~~~~

We recommend to install ChainerMN via :command:`pip`::

  $ pip install chainermn


.. _install-from-source:
  
Install from Source
~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install ChainerMN from source::

  $ tar zxf chainermn-x.y.z.tar.gz
  $ cd chainermn-x.y.z
  $ pip install -r requirements.txt
  $ python setup.py install

.. _non-gpu-env:
  
Non-GPU environments
~~~~~~~~~~~~~~~~~~~~

For users who want to try ChainerMN in a CPU-only environment,
typically for testing for debugging purpose, ChainerMN can be built
with the ``--no-nccl`` flag.::

  $ python setup.py install --no-nccl

In this case, the MPI does not have to be CUDA-aware.
Only ``naive`` communicator works with the CPU mode.
