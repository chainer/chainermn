import chainer
import chainermn
import chainermn.communicators
import chainermn.functions


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
        super(MultiNodeChainGroup, self).add_link(link)
        self._rank_inouts.append((rank_in, rank_out))

    def __call__(self, *inputs):
        x = None
        backward_pointer = None
        n_layers = len(self._children)

        for i, (f, (rank_in, rank_out)) in \
                enumerate(zip(self._children, self._rank_inouts)):
            if rank_in is None:
                x = f(*inputs)
            if rank_in is not None:
                if i == 0:
                    x = chainermn.functions.recv(
                        self._comm,
                        rank=rank_in,
                        device=self._device_id)
                else:
                    # TODO(tsutsumi) is this assertion appropriate?
                    assert backward_pointer is not None
                    x = chainermn.functions.recv_retain(
                        backward_pointer,
                        self._comm,
                        rank=rank_in,
                        device=self._device_id)

                x = f(x)

            if rank_out is None:
                assert i == n_layers - 1, (i, n_layers - 1)
                y = x
            else:
                backward_pointer = chainermn.functions.send(
                    x, self._comm, rank=rank_out)
                y = backward_pointer

        return y
