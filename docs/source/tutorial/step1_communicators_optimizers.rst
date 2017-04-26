Step 1: Communicators and Optimizers
====================================

In the following, we explain how to modify your code using Chainer
to enable distributed training with ChainerMN.
We take `the MNIST example of Chainer <https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py>`_
as an instance and modify it in a step-by-step manner
to see the standard process adopting ChainerMN.


Creating a Communicator
~~~~~~~~~~~~~~~~~~~~~~~

We first need to create a *communicator*.
A communicator is in charge of communication between workers.
A communicator can be created as follows::

  comm = chainermn.create_communicator()


Workers in a node have to use different GPUs.
For this purpose, ``intra_rank`` property of communicators is useful.
Each worker in a node is assigned unique ``intra_rank`` starting from zero.
Therefore, it is often convenient to use the ``intra_rank``-th GPU.

The following is the code line found in the original MNIST example::

  chainer.cuda.get_device(args.gpu).use()

We modify that part as follows::

  device = comm.intra_rank
  chainer.cuda.get_device(device).use()


Creating a Multi-Node Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the most important step.
We need to insert the communication right after backprop
and right before optimization.
In ChainerMN, it is done by creating a *multi-node optimizer*.

Method ``create_multi_node_optimizer`` receives a standard optimizer of Chainer,
and it returns a new optimizer. The returned optimizer is called multi-node optimizer.
It behaves exactly same as the given original standard optimizer
(e.g., you can add hooks such as ``WeightDecay``),
except that it communicates model parameters and gradients properly.

The following is the code line found in the original MNIST example::

  optimizer = chainer.optimizers.Adam()


To obtain a multi-node optimizer, we modify that part as follows::

  optimizer = chainermn.create_multi_node_optimizer(
      chainer.optimizers.Adam(), comm)


Run
~~~

With the above two changes, your script is ready for distributed
training.  Invoke your script with ``mpiexec`` or ``mpirun`` (see your
MPI's manual for details).  The following is an example to execute the
training with four processes at localhost::

  $ mpiexec -n 4 python train_mnist.py


Multi-node execution
~~~~~~~~~~~~~~~~~~~~

If you can successfully run the multi-process version of MNIST
example, you are almost ready for multi-node executions. The simplest
way is to specify ``--host`` argument to :command:`mpiexec`
command. Let's suppose you have two GPU-equipped computing nodes:
``host00`` and ``host01``, so you have 8 GPUs in total::

  $ mpiexec -n 8 -host host00,host01 python train_mnist.py

The script should run identically to the previous intra-node execution.


If you have trouble
~~~~~~~~~~~~~~~~~~~

If you have any trouble running the script in your environment,
go to the :ref:`troubleshooting` page and follow the steps to check
your environment and configuration.

Next Steps
~~~~~~~~~~

Only with the above two changes,
training is already done in a distributed way,
i.e.,
the model parameters are updated
by using gradients that are aggregated over all the workers.
However,
your MNIST example still has a few problems or rooms of improvement.
In the next page, we will see how to address the following problems:

* Training period is wrong; 'one epoch' is not one epoch.
* Evaluation is not parallelized.
* Status outputs to stdout are repeated and annoying.

