import six


class MultiNodeEvaluator(object):

    def __init__(self, actual_evaluator, communicator):
        if hasattr(communicator, 'mpi_comm'):
            communicator = communicator.mpi_comm

        super(MultiNodeEvaluator, self).__setattr__(
            'communicator', communicator)
        super(MultiNodeEvaluator, self).__setattr__(
            'actual_evaluator', actual_evaluator)

        if six.PY2:
            class T(MultiNodeEvaluator, type(actual_evaluator)):
                pass
            self._t = T

    def evaluate(self):
        local_mean_dict = self.actual_evaluator.evaluate()
        global_mean_dict = {
            name: self.communicator.allreduce(value) / self.communicator.size
            for name, value in sorted(local_mean_dict.items())
        }
        return global_mean_dict

    def __getattr__(self, attr_name):
        return getattr(self.actual_evaluator, attr_name)

    def __setattr__(self, attr_name, value):
        if attr_name == '__class__':
            super(MultiNodeEvaluator, self).__setattr__(attr_name, value)
        else:
            setattr(self.actual_evaluator, attr_name, value)

    def __call__(self, *args, **kwargs):
        # TODO(akiba): explain that __call__ is not resolved by __getattr__
        # TODO(akiba): explain that we need to deceive type check of Python 2
        # Problem 1: __call__ unavalable for objects with different types
        #   http://ideone.com/d9S5Gr -- Python 3, standard way works
        #   http://ideone.com/dMXJTS -- Python 2, failure by type check
        #   http://ideone.com/FD7VIv -- Python 2, hack
        # Problem 2: methods of wrapper objects are not called
        #   http://ideone.com/6OCgMb -- Python 3, desired behaviour
        #   http://ideone.com/MobQmY -- Python 2, undesired behaviour
        #   http://ideone.com/e8eI4M -- Python 2, more dirty hack

        if six.PY3:
            type(self.actual_evaluator).__call__(self, *args, **kwargs)
        else:
            t = type(self.actual_evaluator)
            c = self.__class__
            try:
                self.__class__ = self._t
                t.__call__(self, *args, **kwargs)
            finally:
                self.__class__ = c
