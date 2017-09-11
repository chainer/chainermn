import chainermn.functions


class _MultiNodeNStepRNN(object):

    def __init__(self, link, communicator, rank_in, rank_out):
        super(_MultiNodeNStepRNN, self).__setattr__(
            'link', link)
        super(_MultiNodeNStepRNN, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeNStepRNN, self).__setattr__(
            'rank_in', rank_in)
        super(_MultiNodeNStepRNN, self).__setattr__(
            'rank_out', rank_out)

    def __call__(self, *inputs):
        n_cells = self.link.rnn._n_cell
        cells = [None for _ in range(n_cells)]

        if self.rank_in is not None:
            cells = [chainermn.functions.recv(
                self.communicator,
                rank=self.rank_in,
                device=self.link.device)]

        outputs = self.link(*cells, *inputs)
        cells, ys = outputs[:-1], outputs[-1]

        if self.rank_out is not None:
            delegate_variable = None
            cell = cells[0]
            for i in range(len(n_cells)):
                delegate_variable = chainermn.functions.send(
                    cell, self.communicator, rank=self.rank_out)
                if i < len(n_cells) - 1:
                    cell = chainermn.functions.pseudo_connect(
                        delegate_variable, cells[i+1])

            ys = chainermn.functions.pseudo_connect(delegate_variable, ys)

        return ys


def create_multi_node_n_step_rnn(
        link, communicator, rank_in=None, rank_out=None):
    """Create a multi node N-step RNN link from a Chainer N-step RNN link.

    Args:

    Returns:

    """
    return _MultiNodeNStepRNN(link, communicator, rank_in, rank_out)
