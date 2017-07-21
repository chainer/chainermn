import chainer
import chainermn
import chainermn.communicators
import chainermn.functions
import chainermn.functions.point_to_point_communication


class MultiNodeChainGroup(chainer.ChainList):
    """Combining multiple non-connected components of computational graph.

    This class combines each `MultiNodeChain`, which represents one of the
    non-connected component in compuational graph. In `__call__()`,
    the returned object of `MultiNodeChain` (which represents pointer)
    are passed to the next `MultiNodeChain`, in order to retain the
    computational graph connected and make backprop work properly.

    Users add each `MultiNodeChain` by `add_link()` method. Each chain
    is invoked in forward computation according to the order they are added,
    and in backward computation according to the reversed order.
    """

    def __init__(self, comm):
        chainer.utils.experimental('chainermn.MultiNodeChainGroup')
        super(MultiNodeChainGroup, self).__init__()
        self._comm = comm
        self._rank_inouts = []

    def add_link(self, link, rank_in=None, rank_out=None):
        """Register one connected link with its inout rank.

        Args:
            link (chainer.Link): The link object to be registered.
            rank_in (int or list):
                Ranks from which it receives data. If None is specified,
                the model does not receive from any machines.
            rank_out (int or list):
                Ranks to which it sends data. If None is specified,
                the model will not send to any machine.
        """
        super(MultiNodeChainGroup, self).add_link(link)
        self._rank_inouts.append((rank_in, rank_out))

    def __call__(self, *inputs):
        x = None
        y = None
        backward_pointer = None

        for i, (f, (rank_in, rank_out)) in \
                enumerate(zip(self._children, self._rank_inouts)):
            if rank_in is None:
                x = f(*inputs)
            else:
                # Preprocess: receiving inputs from the other machine.
                if isinstance(rank_in, int):
                    rank_in = [rank_in]

                if i == 0:
                    for (j, _rank_in) in enumerate(rank_in):
                        _x = chainermn.functions.recv(
                            self._comm,
                            rank=_rank_in,
                            device=self._device_id)
                        if j == 0:
                            x = _x
                        else:
                            x += _x
                else:
                    # TODO(tsutsumi) is this assertion appropriate?
                    assert backward_pointer is not None

                    x = chainermn.functions.recv_retain(
                        backward_pointer,
                        self._comm,
                        rank=rank_in[0],
                        device=self._device_id)

                    backward_pointer = None

                    for _rank_in in rank_in[1:]:
                        x += chainermn.functions.recv(
                            self._comm,
                            rank=_rank_in,
                            device=self._device_id)

                # Reduce.
                if len(rank_in) > 1:
                    x /= len(rank_in)

                # Actual forward.
                x = f(x)

            if rank_out is None:
                # TODO(tsutsumi) is this assertion appropriate?
                assert y is None
                y = x  # model output
                backward_pointer = y
            else:
                if isinstance(rank_out, int):
                    rank_out = [rank_out]

                if backward_pointer is None:
                    for (j, _rank_out) in enumerate(rank_out):
                        if j == 0:
                            backward_pointer = chainermn.functions.send(
                                x, self._comm, _rank_out)
                        else:
                            backward_pointer = chainermn.functions.send_retain(
                                x, backward_pointer, self._comm, _rank_out)
                else:
                    for _rank_out in rank_out:
                        backward_pointer = chainermn.functions.send_retain(
                            x, backward_pointer, self._comm, _rank_out)

        # Return.
        if y is backward_pointer:
            # The last layer returns model output.
            return y
        elif y is not None:
            # The intermediate layer returns model output.
            return chainermn.functions.point_to_point_communication.merge(
                backward_pointer, y.data)
        else:
            # Do not have any model output.
            return backward_pointer
