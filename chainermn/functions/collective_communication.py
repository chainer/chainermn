import chainer
from chainer import cuda


class AllToAll(chainer.Function):
    """Collective all-to-all communication."""

    def __init__(self, comm, device):
        chainer.utils.experimental('chainermn.functions.AllToAll')
        self.comm = comm
        self.device = device

    def forward(self, inputs):
        if len(inputs) != self.comm.size:
            raise ValueError(
                'The length of inputs must be same as communicator size.')

        xs = tuple([x for x in inputs])
        ys = self.comm.alltoall(xs)

        if isinstance(self.device, int) and self.device >= 0:
            ys = tuple([cuda.to_gpu(y, device=self.device) for y in ys])

        return ys

    def backward(self, inputs, grad_outputs):
        assert self.comm.size == len(grad_outputs)

        xp = cuda.get_array_module(*inputs)
        with cuda.get_device_from_array(*inputs):
            gys = tuple([gy for gy in grad_outputs])
            gx = self.comm.alltoall(gys)
            gx = [xp.array(_gx) for _gx in gx]
            return tuple(gx)


def all_to_all(comm, xs, device=-1):
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
        device (int): Target device specifier.

    Returns:
        ys (list of chainer.Variables): Received variables.
        d: A delegate variable.
    """

    if len(xs) != comm.size:
        raise ValueError('The length of xs must be same as communicator size.')

    return AllToAll(comm, device)(*xs)
