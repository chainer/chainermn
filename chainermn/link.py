import chainer
import chainermn
import chainermn.communicators


class MultiNodeChain(chainer.Chain):
    """Chainer-like composable link interface with distributed communication.

    In the multi-node distributed environment, we need more flexible,
    intuitive interface to define model-parallel neural nets, where inter-node
    communications are highly complicated. Since the original Chainer is
    designed for single-machine computation (or data-parallel with multiple
    GPU, at most), the traditional backprop or garbage collection would easily
    cause unexpected bugs. `MultiNodeChain` is designed to avoid such problems
    peculiar to multi-node computation.

    This class provides a `chainer.Chain` like interface with automatic
    communication under multi-node distributed environment.
    Each `MultiNodeChain` is designed to receive inputs either from the other
    machine or the original dataset, and return outputs which may be sent
    to the other machine. Inter-node communication is not expected to be
    occurred in individual `MultiNodeChain`. If you would like to design
    such a network (i.e., the model whose computational graph is
    non-connected), you shuold first define each connected computational
    graph as one `MultiNodeChain` and use `chainermn.MultiNodeChainGroup` to
    combine them.

    Unlike the original `chainer.Chain`, users must define forward
    computation in `forward()`, in which `chainermn.functions.send` or
    `chainermn.functions.recv` are NOT EXPECTED to be invoked.
    """

    def __init__(self, comm, rank_in=None, rank_out=None, *args, **kwargs):
        """Initialization method of `MultiNodeChain`.

        Args:
            comm (chainermn.communicators._base): ChainerMN communicator.
            rank_in (int):
                MPI rank of the machine from which this model receives inputs.
                The default value is `None`, which means this model would not
                receive inputs from the other machines, rather receive
                the original dataset.
            rank_out (int):
                MPI rank of the machine to which this model sends outputs.
                The default value is `None`, which means this model would not
                send outputs to the other machines.

        """
        chainer.utils.experimental('chainermn.MultiNodeChain')
        assert isinstance(
            comm, chainermn.communicators._base.CommunicatorBase), \
            "comm must be ChainerMN communicator"
        super(MultiNodeChain, self).__init__(*args, **kwargs)
        self._comm = comm
        self._rank_in = rank_in
        self._rank_out = rank_out

    def __call__(self, *args):
        if self._rank_in is None:
            y = self.forward(*args)
        else:
            assert len(args) <= 1, \
                "the number of backward pointer must be less than 1"

            if len(args) == 0:
                x = chainermn.functions.recv(
                    self._comm,
                    rank=self._rank_in,
                    device=self._device_id)
            elif len(args) == 1:
                # `pointer` is a pointer to the former sub-component of
                # computational graph, which is expected to be invoked before
                # this component.
                pointer, = args
                x = chainermn.functions.recv_retain(
                    pointer,
                    self._comm,
                    rank=self._rank_in,
                    device=self._device_id)

            y = self.forward(x)

        if self._rank_out is None:
            return y
        else:
            return chainermn.functions.send(y, self._comm, rank=self._rank_out)

    def forward(self, *args):
        """Forward computation of the model.

        Unlike the original `chainer.Chain`, please define forward computation
        by overriding this method. This method is invoked in
        `__call__`, together with `chainermn.functions.send`
        and `chainermn.functions.recv` if needed.
        """
        raise NotImplementedError()


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
    def __init__(self):
        super(MultiNodeChainGroup, self).__init__()

    def __call__(self, x):
        for f in self.children():
            # Retain the pointer to previous component, connecting to
            # the next component.
            x = f(x)

        return x
