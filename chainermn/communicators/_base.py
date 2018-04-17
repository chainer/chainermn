

class CommunicatorBase(object):
    '''\

    Naming conventions of data transfer methods:
    * Methods that treats Python objects have ``_obj`` suffix
    * Methods that treats Chainer Models have suffix ``_grad``
    * Methods that treats ndarray, or such list have no suffix

    So the number of methods would be::

        [send, recv, bcast, gather, allreduce] * [ '_obj', '']

    (with single exception ``alltoall`` and ``split``,
    ``allreduce_grad`` and ``bcast_data`` so far). Also methods
    are supposed to be written in this order. Methods may also be
    allowed to be left uninmplemented.

    '''

    @property
    def rank(self):
        raise NotImplementedError()

    @property
    def intra_rank(self):
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    def split(self, color, key):
        raise NotImplementedError()

    def alltoall(self, xs):
        raise NotImplementedError()

    # on ndarrays and such
    def send(self, data, dest, tag):
        raise NotImplementedError()

    def recv(self, source, tag):
        raise NotImplementedError()

    def bcast(self, data, max_buf_len=None, root=0):
        raise NotImplementedError()

    def gather(self, data, root=0):
        raise NotImplementedError()

    # TODO(kuenishi) implment ``allreduce``
    # def allreduce(self, xs):

    # on objects
    def send_obj(self, obj, dest, tag):
        raise NotImplementedError()

    def recv_obj(self, source, tag):
        raise NotImplementedError()

    def bcast_obj(self, data, max_buf_len=None, root=0):
        raise NotImplementedError()

    def gather_obj(self, data, root=0):
        raise NotImplementedError()

    # We may add ``op`` argument in future when needed.
    def allreduce_obj(self, obj):
        raise NotImplementedError()

    # Special communication method on grads and data of models
    def bcast_data(self, model, max_buf_len=None, root=0):
        '''Broadcast Chainer model data'''
        raise NotImplementedError()

    # Will be deprecated in some future
    broadcast_data = bcast_data

    def allreduce_grad(self, model):
        raise NotImplementedError()
