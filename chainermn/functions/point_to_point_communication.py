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
        x, = inputs
        self.comm.send(x, self.peer_rank, self.peer_tag)
        return xp.array([]),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        with cuda.get_device_from_array(*inputs):
            gy = self.comm.recv(self.peer_rank, self.peer_tag)
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

        if chainer.__version__.startswith('1.'):
            # For backward compatibility.
            dummy_var = chainer.Variable(xp.array([]), volatile='auto')
        else:
            # This variable is necessary to backprop correctly in Chainer v2.
            # This trick relies on the fact chainer.Variable.requires_grad is
            # True by default at Chainer v2.0.0.
            dummy_var = chainer.Variable(xp.array([]))

        ret = super(Recv, self).__call__(dummy_var)
        return ret

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
        dummy_var = xp.array([[]])
        return dummy_var


def send(x, comm, rank, tag=0):
    """Send elements to target process.

    This function returns a dummy variable only holding the computational
    graph. If ``backward()`` is invoked by this dummy variable, it will
    try to receive gradients from the target process and send them back
    to the parent nodes.

    Args:
        x (Variable): Variable holding a matrix which you would like to send.
        comm (chainer.communicators.CommunicatorBase): ChainerMN communicator.
        rank (int): A number specifying target process.
        tag (int): An optional message ID.

    Returns:
        ~chainer.Variable:
            A dummy variable with no actual data, only holding the
            computational graph. If ``backward()`` is invoked by this dummy
            variable, it will try to receive gradients from the target process.

    """
    return Send(comm, peer_rank=rank, peer_tag=tag)(x)


def recv(comm, rank, tag=0, device=-1):
    """Receive elements from target process.

    This function returns data received from target process. If ``backward()``
    is invoked, it will try to send gradients to the target process.

    Args:
        comm (chainer.communicators.CommunicatorBase): ChainerMN communicator.
        rank (int): A number specifying target process.
        tag (int): An optional message ID.
        device (int): Target device specifier.

    Returns:
        ~chainer.Variable:
            Data received from target process. If ``backward()`` is invoked
            by this variable, it will send gradients to the target process.

    """
    return Recv(comm, peer_rank=rank, peer_tag=tag, device=device)()
