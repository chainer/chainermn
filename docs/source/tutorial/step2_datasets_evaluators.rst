Step 2: Datasets and Evaluators
===============================

Following the previous page, we continue to
explain general steps to modify your code for ChainerMN
through the MNIST example.
All of the steps below are optional,
although useful for many cases.


Scattering Datasets
~~~~~~~~~~~~~~~~~~~

If you want to keep the definition of 'one epoch' correctly,
we need to scatter the dataset to all workers.

For this purpose, ChainerMN provides a method ``scatter_dataset``.
It scatters the dataset of worker 0 (i.e., the worker whose ``comm.rank`` is 0)
to all workers. The given dataset of other workers are ignored.
The dataset is split into sub datasets of almost equal sizes and scattered
to the workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
used.

The following is the code line from the original MNIST example that loads the dataset::

  train, test = chainer.datasets.get_mnist()


We modify it as follows. Only worker 0 loads the dataset, and then it is scattered to all the workers::

  if comm.rank == 0:
      train, test = chainer.datasets.get_mnist()
  else:
      train, test = None, None

  train = chainermn.scatter_dataset(train, comm)
  test = chainermn.scatter_dataset(test, comm)


Replacing Epoch Triggers
~~~~~~~~~~~~~~~~~~~~~~~~

This step is necessary only when you use ``scatter_dataset``.
Please remember that *using normal epoch triggers is dangerous*.
This is because, when the length of the original dataset before scatter
is not divisible by the number of workers,
different workers may have sub datasets of different lengths.
Therefore, epoch triggers may be invoked in different timings,
and this may cause critical problems.

For this purpose, we offer a utility function ``get_epoch_trigger``.
Please note that this function communicates between workers,
so, if you use it, then all the workers should call this.

The following is the code line in the original MNIST example that creates a trainer::

  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


We replace the stop trigger from an epoch trigger to the roughly same interval trigger
by using ``get_epoch_trigger`` as follows::

  trainer = training.Trainer(updater,
      chainermn.get_epoch_trigger(args.epoch, train, args.batchsize, comm),
      out=args.out)



Creating A Multi-Node Evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is also an optional step, but useful when validation is
taking considerable amount of time.
In this case, you can also parallelize the validation by using *multi-node evaluators*.

Similarly to multi-node optimizers, you can create a multi-node evaluator
from a standard evaluator by using method ``create_multi_node_evaluator``.
It behaves exactly same as the given original evaluator
except that it reports the average of results over all workers.

The following line from the original MNIST example adds a evalautor extension to the trainer::

  trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

To create and use a multi-node evaluator, we modify that part as follows::

  evaluator = extensions.Evaluator(test_iter, model, device=device)
  evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
  trainer.extend(evaluator,
        trigger=chainermn.get_epoch_trigger(1, train, args.batchsize, comm))


Suppressing Unnecessary Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of extensions should be invoked only by one of the workers.
For example, if ``PrintReport`` extension is invoked by all of the workers,
many redundant lines will appear in your console.
Therefore, it is convenient to register these extensions
only at workers of rank zero as follows::

  if comm.rank == 0:
      trainer.extend(extensions.dump_graph('main/loss'))
      trainer.extend(extensions.LogReport())
      trainer.extend(extensions.PrintReport(
          ['epoch', 'main/loss', 'validation/main/loss',
           'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
      trainer.extend(extensions.ProgressBar())

Please note that ``chainermn.get_epoch_trigger`` should not be used
for these extensions, even if you use ``scatter_dataset``.  You should
use normal epoch triggers because these extensions are activated only
in one worker.
