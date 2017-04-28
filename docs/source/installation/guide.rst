Installation Guide
==================

Requirements
------------
In addition to Chainer, ChainerMN depends on the following software libraries:
CUDA-Aware MPI, NVIDIA NCCL, and a few Python packages including MPI4py.


Chainer
~~~~~~~

ChainerMN adds distributed training feature to Chainer;
thus it naturally requires Chainer.
Please refer to `the official instructions <http://docs.chainer.org/en/latest/install.html>`_ to install.



.. _mpi-install:

CUDA-Aware MPI
~~~~~~~~~~~~~~

ChainerMN relies on MPI.
In particular, for efficient communication between GPUs, it uses CUDA-aware MPI.
About CUDA-aware MPI, see `this introduction article <https://devblogs.nvidia.com/parallelforall/introduction-cuda-aware-mpi/>`_.

Several MPI packages support the CUDA-aware feature.
They generally require to be configured and built properly.
The following are examples of MVAPICH and OpenMPI.


MVAPICH (for details, see `the official instruction <http://mvapich.cse.ohio-state.edu/static/media/mvapich/mvapich2-2.0-userguide.html#x1-120004.5>`_)::

  $ ./configure --enable-cuda
  $ make -j4
  $ sudo make install
  $ export MV2_USE_CUDA=1  # Should be set all the time when using ChainerMN

OpenMPI (for details, see `the official instruction <https://www.open-mpi.org/faq/?category=building#build-cuda>`_)::

  $ ./configure --with-cuda
  $ make -j4
  $ sudo make install


.. _nccl-install:
  
NVIDIA NCCL
~~~~~~~~~~~

To enable efficient intra-node GPU-to-GPU communication,
we use `NVIDIA NCCL <https://github.com/NVIDIA/nccl>`_.
See `the official instructions <https://github.com/NVIDIA/nccl#build--run>`_ for installation.

ChainerMN requires NCCL even if you have only one GPU per node.  The
only exception is when you run ChainerMN on CPU-only environments. See
:ref:`non-gpu-env` for more details.

.. _mpi4py-install:

MPI4py
~~~~~~

ChainerMN depends on a few Python packages, which are listed in ``requirements.txt``.
They are automatically installed when you install ChainerMN via PyPI.

However, among them, we need to be a little careful about MPI4py.
It links to MPI at the installation time, so please be sure
to properly configure environment variables
so that MPI is available at the installation time.
In particular, if you have multiple MPI implementations in your environment,
please expose the implementation that you want to use
both when you install and use ChainerMN.

.. note::

  If you are not using GPUs, communicator ``naive`` works with *non*-CUDA-aware MPI.

  Communicators ``naive`` and ``flat`` can be used without NCCL.
  However, it is common that they are far slower than other communicators
  on environment where a node contains multiple GPUs.


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

For users who wnat to try ChainerMN in a CPU-only environment,
typically for testing for debuggin purpose, ChainerMN can be build
with ``--no-nccl`` flag.::

  $ python setup.py install --no-nccl

In this case, the MPI does not have to be CUDA-aware.
