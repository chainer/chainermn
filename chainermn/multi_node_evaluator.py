class MultiNodeEvaluator(object):

    def __init__(self, actual_evaluator, communicator):
        if hasattr(communicator, 'mpi_comm'):
            communicator = communicator.mpi_comm

        super(MultiNodeEvaluator, self).__setattr__(
            'communicator', communicator)
        super(MultiNodeEvaluator, self).__setattr__(
            'actual_evaluator', actual_evaluator)

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
        setattr(self.actual_evaluator, attr_name, value)

    def __call__(self, *args, **kwargs):
        # TODO(akiba): explain that __call__ is not resolved by __getattr__
        type(self.actual_evaluator).__call__(self, *args, **kwargs)
