Install Guide
=============

Requirements
------------
In addition to Chainer, ChainerMN depends on the following software libraries:
CUDA-Aware MPI, NVIDIA NCCL, and MPI4py.


Chainer
~~~~~~~

ChainerMN adds distributed training feature to Chainer,
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


NVIDIA NCCL
~~~~~~~~~~~

To enable efficient intra-node GPU-to-GPU communication,
we use `NVIDIA NCCL <https://github.com/NVIDIA/nccl>`_.
See `the official instructions <https://github.com/NVIDIA/nccl#build--run>`_ for installation.


MPI4py
~~~~~~

ChainerMN depends on MPI4py. It can be installed via ``pip`` as follows::

  $ pip install mpi4py


.. note::

  If you are not using GPUs, communicator ``naive`` works with *non*-CUDA-aware MPI.

  Communicators ``naive`` and ``flat`` can be used without NCCL.
  However, it is common that they are far slower than other communicators
  on environment where a node contains multiple GPUs.


Installation
------------

Install via pip
~~~~~~~~~~~~~~~

We recommend to install ChainerMN via pip::

  $ pip install chainermn


Install from Source
~~~~~~~~~~~~~~~~~~~

You can use ``setup.py`` to install ChainerMN from source::

  $ tar zxf chainermn-x.y.z.tar.gz
  $ cd chainermn-x.y.z
  $ python setup.py install

When your environment does not have NCCL, pass ``--no-nccl`` flag to ``setup.py``::

  $ python setup.py install --no-nccl

