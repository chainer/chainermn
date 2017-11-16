import chainer


class Alltoall(chainer.Function):
    """Collective all-to-all communication."""

    def __init__(self, comm):
        chainer.utils.experimental('chainermn.functions.Alltoall')
        self.comm = comm

    def forward(self, inputs):
        if len(inputs) != self.comm.size:
            raise ValueError(
                'The length of inputs must be same as communicator size.')

        xs = tuple([x for x in inputs])
        ys = self.comm.alltoall(xs)
        return ys

    def backward(self, inputs, grad_outputs):
        assert self.comm.size == len(grad_outputs)
        gys = tuple([gy for gy in grad_outputs])
        gx = self.comm.alltoall(gys)
        return gx


def all_to_all(comm, xs):
    """Differentiable all-to-all communication between workers.

    This function invokes all-to-all communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, just passing input gradients back.
    Unlike point-to-point communication such as ``chainermn.functions.send``
    and ``chainermn.functions.recv``, users need not to care about
    delegate variables, since ``backward()`` will not be invoked until
    all gradients from output direction arrive.
    Please refer to ``chainermn.functions.pseudo_connect`` about the detail
    of delegate variables.

    Args:
        comm: ChainerMN communicator.
        xs (list of chainer.Variables): Variables to send.

    Returns:
        ys (list of chainer.Variables): Received variables.
        d: A delegate variable.
    """

    if len(xs) != comm.size:
        raise ValueError('The length of xs must be same as communicator size.')

    return Alltoall(comm)(*xs)
