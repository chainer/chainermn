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


class Bcast(chainer.Function):
    """Collective broadcast communication."""

    def __init__(self, comm, root, device):
        chainer.utils.experimental('chainermn.functions.Bcast')
        self.comm = comm
        self.root = root
        self.device = device

    def forward(self, inputs):
        if self.comm.rank == self.root:
            x, = inputs
        else:
            x = None
        x = self.comm.bcast(x, self.root)

        if isinstance(self.device, int) and self.device >= 0:
            x = cuda.to_gpu(x, device=self.device)

        return x,

    def backward(self, inputs, grad_outputs):
        with cuda.get_device_from_array(*inputs):
            gx, = grad_outputs
            gxs = self.comm.gather(gx, self.root)

            if self.comm.rank == self.root:
                if isinstance(self.device, int) and self.device >= 0:
                    gxs = cuda.to_gpu(gxs, device=self.device)

                return gxs.sum(axis=0),
            else:
                return None,


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
    """
    chainer.utils.experimental('chainermn.functions.all_to_all')

    if len(xs) != comm.size:
        raise ValueError('The length of xs must be same as communicator size.')

    return AllToAll(comm, device)(*xs)


def bcast(comm, x, root=0, device=-1):
    """Differentiable broadcast communication between workers.

    This function invokes broadcast communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are gathered to the root process
    and summed up.

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variable): Variable to be sent.
        device (int): Target device specifier.

    Returns:
        y (chainer.Variable): Broadcasted variable.
    """
    chainer.utils.experimental('chainermn.functions.bcast')

    return Bcast(comm, root, device)(x)
