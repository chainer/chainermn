import chainer
import chainer.functions.connection as conn
import chainermn.functions


_rnn_n_cells = {
    conn.n_step_gru.n_step_bigru: 1,
    conn.n_step_gru.n_step_gru: 1,
    conn.n_step_lstm.n_step_bilstm: 2,
    conn.n_step_lstm.n_step_lstm: 2,
    conn.n_step_rnn.n_step_birnn: 1,
    conn.n_step_rnn.n_step_rnn: 1,
}


class _MultiNodeNStepRNN(chainer.Chain):

    def __init__(self, link, communicator, rank_in, rank_out):
        super(_MultiNodeNStepRNN, self).__init__(actual_rnn=link)

        self.communicator = communicator
        self.rank_in = rank_in
        self.rank_out = rank_out

        if not hasattr(link, 'rnn') or link.rnn not in _rnn_n_cells:
            raise ValueError('link must be NStepRNN and its inherited link')
        else:
            self.n_cells = _rnn_n_cells[link.rnn]

    def __call__(self, *inputs):
        cells = [None for _ in range(self.n_cells)]

        if self.rank_in is not None:
            cells = [chainermn.functions.recv(
                self.communicator,
                rank=self.rank_in,
                device=self.actual_rnn._device_id)
                for _ in range(self.n_cells)]

        outputs = self.actual_rnn(*(tuple(cells) + inputs))
        cells, ys = outputs[:-1], outputs[-1]

        delegate_variable = None
        if self.rank_out is not None:
            cell = cells[0]
            for i in range(self.n_cells):
                delegate_variable = chainermn.functions.send(
                    cell, self.communicator, rank=self.rank_out)
                if i < self.n_cells - 1:
                    cell = chainermn.functions.pseudo_connect(
                        delegate_variable, cells[i + 1])

        return cells + tuple([ys, delegate_variable])


def create_multi_node_n_step_rnn(
        link, communicator, rank_in=None, rank_out=None):
    """Create a multi node N-step RNN link from a Chainer N-step RNN link.

    Args:

    Returns:

    """
    return _MultiNodeNStepRNN(link, communicator, rank_in, rank_out)
