import chainer
from chainer import cuda
import numpy


class AllGather(chainer.Function):
    """Collective all-gather communication."""

    def __init__(self, comm, device):
        chainer.utils.experimental('chainermn.functions.AllGather')
        self.comm = comm
        self.device = device

    def forward(self, inputs):
        x, = inputs
        ys = self.comm.allgather(x)

        if isinstance(self.device, int) and self.device >= 0:
            ys = tuple([cuda.to_gpu(y, device=self.device) for y in ys])

        return ys

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gxs = self.comm.alltoall(grad_outputs)

        if isinstance(self.device, int) and self.device >= 0:
            gxs = tuple([cuda.to_gpu(gx, device=self.device) for gx in gxs])

        gx = xp.stack(gxs).sum(axis=0)
        return gx,


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

    def __call__(self, *inputs):
        xp = cuda.get_array_module(*inputs)

        if inputs == ():
            # Without dummy variable, this function does not "require_grad",
            # thus back propagation will not be invoked.
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Bcast, self).__call__(dummy_var)

        else:
            return super(Bcast, self).__call__(*inputs)

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
                gxs = numpy.stack(gxs)

                if isinstance(self.device, int) and self.device >= 0:
                    gxs = cuda.to_gpu(gxs, device=self.device)

                return gxs.sum(axis=0),
            else:
                return None,


class Gather(chainer.Function):
    """Collective gather communication."""

    def __init__(self, comm, root, device):
        chainer.utils.experimental('chainermn.functions.Gather')
        self.comm = comm
        self.root = root
        self.device = device

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        ys = self.comm.gather(x, self.root)

        if self.comm.rank == self.root:
            if isinstance(self.device, int) and self.device >= 0:
                ys = tuple([cuda.to_gpu(y, device=self.device) for y in ys])

            return ys

        else:
            # Return an empty variable, which serves as "delegate_variable."
            return xp.array([], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        with cuda.get_device_from_array(*inputs):
            gx = self.comm.scatter(grad_outputs, self.root)

            if isinstance(self.device, int) and self.device >= 0:
                gx = cuda.to_gpu(gx, device=self.device)

            return gx,


class Scatter(chainer.Function):
    """Collective scatter communication."""

    def __init__(self, comm, root, device):
        chainer.utils.experimental('chainermn.functions.Scatter')
        self.comm = comm
        self.root = root
        self.device = device

    def __call__(self, *inputs):
        xp = cuda.get_array_module(*inputs)

        if inputs == ():
            # Without dummy variable, this function does not "require_grad",
            # thus back propagation will not be invoked.
            dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))
            dummy_var.name = 'dummy_var'
            return super(Scatter, self).__call__(dummy_var)

        else:
            return super(Scatter, self).__call__(*inputs)

    def forward(self, inputs):
        if self.comm.rank == self.root:
            y = self.comm.scatter(inputs, self.root)
        else:
            y = self.comm.scatter(None, self.root)

        if isinstance(self.device, int) and self.device >= 0:
            y = cuda.to_gpu(y, device=self.device)

        return y,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        with cuda.get_device_from_array(*inputs):
            gy, = grad_outputs
            gxs = self.comm.gather(gy, self.root)

            if self.comm.rank == self.root:
                if isinstance(self.device, int) and self.device >= 0:
                    gxs = tuple([cuda.to_gpu(gx, device=self.device)
                                 for gx in gxs])

                return gxs

            else:
                # Slave processes need to maintain input/output shapes.
                if inputs == ():
                    dummy_var = tuple([xp.array([], dtype=xp.float32)])
                else:
                    dummy_var = tuple([xp.zeros(x.shape, dtype=xp.float32)
                                       for x in inputs])
                return dummy_var


def allgather(comm, x, device=-1):
    """Differentiable all-gather communication between workers.

    This function invokes gather communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are reduced to each process.

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variables): Variables to send.
        device (int): Target device specifier.

    Returns:
        ys (list of chainer.Variables): Received variables.
    """
    chainer.utils.experimental('chainermn.functions.all_gather')

    return AllGather(comm, device)(x)


def alltoall(comm, xs, device=-1):
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

    if comm.rank == root:
        return Bcast(comm, root, device)(x)
    else:
        return Bcast(comm, root, device)()


def gather(comm, x, root=0, device=-1):
    """Differentiable gather communication between workers.

    This function invokes gather communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are scattered from the root process
    to each slave.

    Args:
        comm: ChainerMN communicator.
        x (chainer.Variable): Variable to be sent.
        device (int): Target device specifier.

    Returns:
        ys (chainer.Variable):
            Gathered variables. ``None`` for slaves.
    """
    chainer.utils.experimental('chainermn.functions.gather')

    return Gather(comm, root, device)(x)


def scatter(comm, xs, root=0, device=-1):
    """Differentiable scatter communication between workers.

    This function invokes scatter communications among processes specified
    by the communicator. Backward will be invoked as well as the ordinary
    chainer functions, where gradients are gathered to the root process.

    Args:
        comm: ChainerMN communicator.
        xs (list of chainer.Variable):
            Variables to be scattered for master process.
            ``None`` for slave process.
        device (int): Target device specifier.

    Returns:
        y (chainer.Variable): Scattered variable.
    """
    chainer.utils.experimental('chainermn.functions.scatter')

    if comm.rank == root:
        return Scatter(comm, root, device)(*xs)
    else:
        return Scatter(comm, root, device)()
