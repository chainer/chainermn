import types


def create_multi_node_evaluator(evaluator, communicator):
    """Create a multi node evaluator from a normal evaluator.

    Actually patches the evaluator to work in multinode
    environment. This method adds several hidden attributes starting
    with _mn_ prefix. After the patch, original evaluator does not
    work correctly in non-MPI environment.

    Args:
        evaluator: evaluator to be patched
            (e.g., ``chainer.training.extensions.Evaluator``)
        communicator: ChainerMN communicator

    Returns:
        The multi node evaluator based on ``evaluator``.

    """

    evaluator._mn_original_evaluate = evaluator.evaluate
    evaluator._mn_communicator = communicator

    def new_evaluate(self):
        local_mean_dict = self._mn_original_evaluate()
        global_mean_dict = {
            name:
            self._mn_communicator.mpi_comm.allreduce(
                value) / self._mn_communicator.size
            for name, value in sorted(local_mean_dict.items())
        }
        return global_mean_dict

    evaluator.evaluate = types.MethodType(new_evaluate, evaluator)
    return evaluator
