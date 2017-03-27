class _MultiNodeOptimizer(object):

    def __init__(self, actual_optimizer, communicator):
        super(_MultiNodeOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MultiNodeOptimizer, self).__setattr__(
            'needs_broadcast', True)

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward()
            del loss

        if self.needs_broadcast:
            self.communicator.broadcast_data(target)
            super(_MultiNodeOptimizer, self).__setattr__(
                'needs_broadcast', False)
        else:
            self.communicator.allreduce_grad(target)
            self.actual_optimizer.update(None, *args, **kwds)

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


def create_multi_node_optimizer(actual_optimizer, communicator):
    return _MultiNodeOptimizer(actual_optimizer, communicator)
