def create_multi_node_evaluator(actual_evaluator, communicator):
    """Create a multi node evaluator from a normal evaluator.

    Args:
        actual_evaluator: evaluator
            (e.g., ``chainer.training.extensions.Evaluator``)
        communicator: ChainerMN communicator

    Returns:
        The multi node evaluator based on ``actual_evaluator``.

    """

    class MultiNodeEvaluator(type(actual_evaluator)):

        def __init__(self, actual_evaluator_, communicator_):
            if hasattr(communicator_, 'mpi_comm'):
                communicator_ = communicator_.mpi_comm

            super(MultiNodeEvaluator, self).__setattr__(
                'communicator', communicator_)
            super(MultiNodeEvaluator, self).__setattr__(
                'actual_evaluator', actual_evaluator_)

        def __getattr__(self, attr_name):
            return getattr(self.actual_evaluator, attr_name)

        def __setattr__(self, attr_name, value):
            setattr(self.actual_evaluator, attr_name, value)

        def evaluate(self):
            local_mean_dict = self.actual_evaluator.evaluate()
            global_mean_dict = {
                name:
                    self.communicator.allreduce(value) / self.communicator.size
                for name, value in sorted(local_mean_dict.items())
            }
            return global_mean_dict

    return MultiNodeEvaluator(actual_evaluator, communicator)
