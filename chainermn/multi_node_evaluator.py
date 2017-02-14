import chainer.training.extensions


class MultiNodeEvaluator(chainer.training.extensions.Evaluator):

    def __init__(self, comm, *args, **kwargs):
        super(MultiNodeEvaluator, self).__init__(*args, **kwargs)

        self.comm = comm
        if hasattr(self.comm, 'mpi_comm'):
            self.comm = self.comm.mpi_comm

    def evaluate(self):
        local_mean_dict = super(MultiNodeEvaluator, self).evaluate()
        global_mean_dict = {
            name: self.comm.allreduce(value) / self.comm.size
            for name, value in sorted(local_mean_dict.items())
        }
        return global_mean_dict
