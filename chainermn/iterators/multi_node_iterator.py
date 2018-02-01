import chainer
import chainermn
import numpy


class _MultiNodeIterator_Master(object):

    def __init__(self, actual_iterator, communicator, rank_master, device):
        self.communicator = communicator
        self.rank_master = rank_master
        self.actual_iterator = actual_iterator
        self.device = device

    def __next__(self):
        try:
            batch = self.actual_iterator.__next__()
            stop = False
        except StopIteration:
            stop = True

        # Broadcast whether stop signal is received before broadcasting data.
        # TODO(tsutsumi): should we prepare API to broadcast flag?
        _stop = numpy.ones((1, ), dtype=numpy.float32) * int(stop)
        self.communicator.bcast(_stop, root=self.rank_master)

        if not stop:
            if isinstance(batch, list):
                batch = numpy.array(batch)
            return chainermn.functions.bcast(
                self.communicator, batch, self.rank_master, self.device)
        else:
            raise StopIteration

    next = __next__

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr_(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)


class _MultiNodeIterator_Slave(chainer.dataset.iterator.Iterator):

    def __init__(self, communicator, rank_master, device):
        super(_MultiNodeIterator_Slave, self).__init__()
        self.communicator = communicator
        self.rank_master = rank_master
        self.device = device

    def __next__(self):
        # Check if master iterator received stop signal.
        stop = None
        stop = self.communicator.bcast(stop, root=self.rank_master)

        if not int(stop):
            return chainermn.functions.bcast(
                self.communicator, None, self.rank_master, self.device)
        else:
            raise StopIteration


def create_multi_node_iterator(
        actual_iterator, communicator, rank_master=0, device=-1):
    """Create a multi node iterator from a Chainer iterator.

    This iterator shares the same batches on multiple processes, simply
    broadcasting batches from master process to slave processes
    in each iteration.
    Master process obtains batches from ``actual_iterator``, which you can
    specify any Chainer iterator (e.g. ``chainer.iterators.SerialIterator``).

    Here is an example situation. When we train a sequence-to-sequence model,
    where the encoder and the decoder is located on two different processes,
    we want to share the same batches on each process, thus inputs for
    the encoder and output teacher signals for the decoder become consistent.

    In order to use the multi node iterator, first create the iterator
    from Chainer iterator and ChainerMN communicator::

        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                dataset, batch_size, shuffle=True),
            communicator)

    Then you can use it as the ordinary Chainer iterator::

        updater = chainer.training.StandardUpdater(iterator, optimizer)
        trainer = training.Trainer(updater)
        trainer.run()

    Since this iterator shares batches through network in each iteration,
    communication might be large. If you train your model-parallel network
    on extremely large dataset, you can also consider to use
    ``chainermn.iterators.create_synchronized_iterator``.

    Args:
        actual_iterator: Chainer iterator
            (e.g., ``chainer.iterators.SerialIterator``).
        communicator: ChainerMN communicator.
        rank_master: process rank to be master.
        device: Target device specifier.

    Returns:
        The master-slave iterator based on ``actual_iterator``.
    """
    if communicator.rank == rank_master:
        return _MultiNodeIterator_Master(
            actual_iterator, communicator, rank_master, device)
    else:
        return _MultiNodeIterator_Slave(communicator, rank_master, device)
