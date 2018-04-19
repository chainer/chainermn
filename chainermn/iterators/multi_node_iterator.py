import chainer
import numpy


class _MultiNodeIterator_Master(object):

    def __init__(self, actual_iterator, communicator, rank_master):
        self.communicator = communicator
        self.rank_master = rank_master
        self.actual_iterator = actual_iterator

        _dataset_size = numpy.ones((1, )).astype(numpy.float32) \
            * len(self.actual_iterator.dataset)
        # TODO(tsutsumi): potential deadlock?
        self.communicator.bcast(_dataset_size, root=self.rank_master)
        self.communicator.bcast(
            self.actual_iterator._order.astype(numpy.float32),
            root=self.rank_master)

    def __next__(self):
        try:
            batch = self.actual_iterator.__next__()
            stop = False
        except StopIteration:
            stop = True
        is_new_epoch = self.actual_iterator.is_new_epoch

        # Notify the followings to slave iterators:
        # 1. whether stop signal is received before broadcasting data.
        # 2. is_new_epoch.
        # 3. current_position.
        _info = numpy.ones((3, )) \
            * [int(stop),
               int(is_new_epoch),
               int(self.actual_iterator.current_position)]
        _info = _info.astype(numpy.float32)
        self.communicator.bcast(_info, root=self.rank_master)

        if not stop:
            if isinstance(batch, list):
                batch = numpy.array(batch)
            batch = self.communicator.bcast(batch, root=self.rank_master)
            return batch.tolist()
        else:
            raise StopIteration

    next = __next__

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr_(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)

    def serialize(self, serializer):
        # Master's and Slave's serialize must be called at the same time.
        self.actual_iterator.serialize(serializer)
        self.communicator.bcast_obj(
            serializer, root=self.rank_master)


class _MultiNodeIterator_Slave(chainer.dataset.iterator.Iterator):

    def __init__(self, communicator, rank_master):
        super(_MultiNodeIterator_Slave, self).__init__()
        self.communicator = communicator
        self.rank_master = rank_master

        # Compatibility to Chainer iterators.
        self.epoch = 0
        self.current_position = 0
        self.is_new_epoch = False

        # TODO(tsutsumi): potential deadlock?
        _size = self.communicator.bcast(None, root=self.rank_master)
        self.dataset_size = int(_size)
        self._order = self.communicator.bcast(None, root=self.rank_master)
        self._order = self._order.astype(numpy.int64)

    def __next__(self):
        # Check if master iterator received stop signal.
        _info = self.communicator.bcast(None, root=self.rank_master)
        stop = bool(_info[0])
        self.is_new_epoch = bool(_info[1])
        self.current_position = int(_info[2])

        if self.is_new_epoch:
            self.epoch += 1

        if not stop:
            batch = self.communicator.bcast(None, root=self.rank_master)
            return batch.tolist()
        else:
            raise StopIteration

    @property
    def epoch_detail(self):
        return self.epoch + 1. * self.current_position / self.dataset_size

    def serialize(self, serializer):
        # Master's and Slave's serialize must be called at the same time.
        _serializer = self.communicator.bcast_obj(
            None, root=self.rank_master)

        self.current_position = serializer(
            'current_position',
            _serializer('current_position', self.current_position)
        )
        self.epoch = serializer('epoch', _serializer('epoch', self.epoch))
        self.is_new_epoch = serializer(
            'is_new_epoch',
            _serializer('is_new_epoch', self.is_new_epoch)
        )

        try:
            self._order = serializer(
                'order',
                _serializer('order', self._order)
            )
        except KeyError:
            pass


def create_multi_node_iterator(
        actual_iterator, communicator, rank_master=0):
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

    .. note:: ``create_multi_node_iterator`` and ``serialize`` of created
              iterators must be called at the same time by master and slaves,
              unless it falls into deadlock because they synchronize internal
              states of iterators.

    Args:
        actual_iterator: Chainer iterator
            (``chainer.iterators.SerialIterator`` and
            ``chainer.iterators.MultiprocessIterator`` are supported).
        communicator: ChainerMN communicator.
        rank_master: process rank to be master.

    Returns:
        The master-slave iterator based on ``actual_iterator``.
    """
    chainer.utils.experimental(
        'chainermn.iterators.create_multi_node_iterator')
    if communicator.rank == rank_master:
        return _MultiNodeIterator_Master(
            actual_iterator, communicator, rank_master)
    else:
        return _MultiNodeIterator_Slave(communicator, rank_master)
