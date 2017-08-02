import chainer
from chainer import cuda
import chainer.utils


class Send(chainer.Function):
    """Send elements to target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        chainer.utils.experimental('chainermn.functions.Send')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        # Note: inputs[1] might contain backward_pointer.
        x = inputs[0]
        self.comm.send(x, self.peer_rank, self.peer_tag)
        # Return an empty variable, which serves as "backward_pointer."
        return xp.array([], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        with cuda.get_device_from_array(*inputs):
            gy = self.comm.recv(self.peer_rank, self.peer_tag)
            if len(inputs) > 1:
                # Dummy grad for backward_pointer.
                # This grad will not be used, only for silencing type checker.
                grad_backward_pointer = inputs[1]
                return xp.array(gy), grad_backward_pointer
            else:
                return xp.array(gy),


class Recv(chainer.Function):
    """Receive elements from target process."""

    def __init__(self, comm, peer_rank, peer_tag, device=-1):
        chainer.utils.experimental('chainermn.functions.Recv')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag
        self.device = device

    def __call__(self, *inputs):
        xp = cuda.get_array_module(*inputs)

        if inputs == ():
            # Expected to be invoked without any args in usual case.
            if chainer.__version__.startswith('1.'):
                # For backward compatibility.
                dummy_var = chainer.Variable(
                    xp.array([], dtype=xp.float32),
                    volatile='auto')
            else:
                # This variable is necessary to backprop correctly
                # in Chainer v2. This trick relies on the fact
                # chainer.Variable.requires_grad is True by default
                # in Chainer v2.0.0.
                dummy_var = chainer.Variable(xp.array([], dtype=xp.float32))

            return super(Recv, self).__call__(dummy_var)

        else:
            # Used for retaining computational graph.
            return super(Recv, self).__call__(*inputs)

    def forward(self, inputs):
        x = self.comm.recv(self.peer_rank, self.peer_tag)
        if isinstance(self.device, int) and self.device >= 0:
            return cuda.to_gpu(x, device=self.device),
        else:
            return x,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gw, = grad_outputs
        self.comm.send(gw, self.peer_rank, self.peer_tag)
        dummy_var = xp.array([[]], dtype=xp.float32)
        return dummy_var


class PseudoConnect(chainer.Function):
    """Connect a variable with backward pointer."""

    def __init__(self, actual_values):
        self._actual_values = actual_values

    def forward(self, inputs):
        return self._actual_values

    def backward(self, inputs, grad_outputs):
        # Inputs of PseudoConnect (i.e., backward_pointer) do not need
        # backward gradients, instead taking consistency of shapes of grads.
        return inputs


def send(x, communicator, rank, tag=0):
    """Send elements to target process.

    This function returns a dummy variable only holding the computational
    graph. If ``backward()`` is invoked by this dummy variable, it will
    try to receive gradients from the target process and send them back
    to the parent nodes.

    Args:
        x (Variable): Variable holding a matrix which you would like to send.
        communicator (chainer.communicators.CommunicatorBase):
            ChainerMN communicator.
        rank (int): Target process specifier.
        tag (int): Optional message ID (MPI feature).

    Returns:
        ~chainer.Variable:
            A dummy variable with no actual data, only holding the
            computational graph. We call this backward_pointer.
            If ``backward()`` is invoked by backward_pointer,
            it will try to receive gradients from the target process.

    """
    chainer.utils.experimental('chainermn.functions.send')
    return Send(communicator, peer_rank=rank, peer_tag=tag)(x)


def recv(communicator, rank, backward_pointer=None, tag=0, device=-1):
    """Receive elements from target process.

    This function returns data received from target process. If ``backward()``
    is invoked, it will try to send gradients to the target process.

    .. note::
        If you define non-connected computational graph on one machine,
        you have to use ``backward_pointer`` to specify the output of
        previous computational graph component.
        Otherwise ``backward()`` does not work well.

    Args:
        communicator (chainer.communicators.CommunicatorBase):
            ChainerMN communicator.
        rank (int): Target process specifier.
        backward_pointer (chainer.Variable):
            Pointer to the other non-connected component.
        tag (int): Optional message ID (MPI feature).
        device (int): Target device specifier.

    Returns:
        ~chainer.Variable:
            Data received from target process. If ``backward()`` is invoked
            by this variable, it will send gradients to the target process.

    """
    chainer.utils.experimental('chainermn.functions.recv')
    if backward_pointer is None:
        return Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag,
            device=device)()
    else:
        return Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag,
            device=device)(backward_pointer)


def pseudo_connect(backward_pointer, *actual_values):
    """Connect independent connected graph component.

    In model-parallel framework, models sometimes have many non-connected
    components. When some additional components follow model outputs,
    outputs of the last component must be merged with model outputs.
    Otherwise backprop does not work well, got stuck into dead lock.

    Args:
        backward_pointer (chainer.Variable):
            Pointer to the previous non-connected graph component.
        actual_values (tuple of numpy.ndarray):
            Actual values which ``backward_pointer`` imitate.

    Returns:
        ~chainer.Variable:
            A variable with the given values combined with backward pointer.
    """
    chainer.utils.experimental('chainermn.functions.pseudo_connect')
    return PseudoConnect(actual_values)(backward_pointer)
