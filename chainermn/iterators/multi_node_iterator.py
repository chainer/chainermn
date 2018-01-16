import chainer
import chainermn


class _MultiNodeIterator_Master(object):

    def __init__(self, actual_iterator, communicator, rank_master, device):
        self.communicator = communicator
        self.rank_master = rank_master
        self.actual_iterator = actual_iterator
        self.device = device

    def __next__(self):
        batch = self.actual_iterator.__next__()
        return chainermn.functions.bcast(
            self.communicator, batch, self.rank_master, self.device)

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr_(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)


class _MultiNodeIterator_Slave(chainer.iterators.Iterator):

    def __init__(self, communicator, rank_master, device):
        super(_MultiNodeIterator_Slave, self).__init__()
        self.communicator = communicator
        self.rank_master = rank_master
        self.device = device

    def __next__(self):
        return chainermn.functions.bcast(
            self.communicator, None, self.rank_master, self.device)


def create_multi_node_iterator(
        actual_iterator, communicator, rank_master=0, device=-1):
    """Create a multi node iterator from a Chainer iterator.

    This is used when you want to broadcast batches from a master process
    to slave processes in each iteration.
    The master process uses ``actual_iterator`` and sharing obtained batches
    to slave processes at the same time.

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
