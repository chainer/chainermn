import collections

import chainer
from chainer import cuda
import chainer.utils
import chainermn.functions


class Send(chainer.Function):
    """Send elements to target process."""

    def __init__(self, comm, peer_rank, peer_tag):
        chainer.utils.experimental('chainermn.functions.Send')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag

    @property
    def label(self):
        return "{} (peer_rank: {})".format(
            self.__class__.__name__,
            self.peer_rank)

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        # TODO(tsutsumi): if len(inputs) == 1, should we unpack inputs?
        self.comm.send(inputs, self.peer_rank, self.peer_tag)

        # Return an empty variable, which serves as "delegate_variable."
        return xp.array([], dtype=xp.float32),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        with cuda.get_device_from_array(*inputs):
            grad = self.comm.recv(self.peer_rank, self.peer_tag)
            if isinstance(grad, tuple):
                return tuple([xp.array(gy) for gy in grad])
            else:
                return xp.array(grad),


class Recv(chainer.Function):
    """Receive elements from target process."""

    def __init__(self, comm, peer_rank, peer_tag, device=-1):
        chainer.utils.experimental('chainermn.functions.Recv')
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag
        self.device = device

    def __call__(self, *inputs):
        data = self.comm.recv(self.peer_rank, self.peer_tag)

        if chainer.__version__.startswith('1.'):
            # For backward compatibility.
            if isinstance(data, tuple):
                var = [chainer.Variable(x, volatile='auto') for x in data]
            else:
                var = [chainer.Variable(data, volatle='auto')]
        else:
            if isinstance(data, tuple):
                var = [chainer.Variable(x) for x in data]
            else:
                var = [chainer.Variable(data)]

        if inputs != ():
            assert len(inputs) == 1, \
                'No more than one delegate_variable will be passed.'

            # inputs might contain delegate_variable.
            # In this case, we must assure the backward path.
            delegate_variable = inputs[0]
            var[0] = chainermn.functions.pseudo_connect(
                delegate_variable, var[0])

        for x in var:
            x.name = 'received_data'

        return super(Recv, self).__call__(*var)

    @property
    def label(self):
        return "{} (peer_rank: {})".format(
            self.__class__.__name__,
            self.peer_rank)

    def forward(self, inputs):
        # Simply return the received data (passed via inputs).
        if isinstance(self.device, int) and self.device >= 0:
            return tuple([cuda.to_gpu(x, device=self.device) for x in inputs])
        else:
            return inputs

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        self.comm.send(grad_outputs, self.peer_rank, self.peer_tag)

        # dummy_var is needed to maintain Chainer's constraint.
        dummy_var = tuple([xp.zeros(x.shape, dtype=xp.float32) for x in inputs])

        return dummy_var


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
            computational graph. Please refer
            ``chainermn.functions.pseudo_connect`` for detail.

    """
    chainer.utils.experimental('chainermn.functions.send')

    if rank == communicator.rank:
        raise ValueError(
            'rank must be different from communicator rank, '
            'otherwise deadlock occurs')

    if isinstance(x, collections.Iterable):
        delegate_variable = Send(communicator, peer_rank=rank, peer_tag=tag)(*x)
    else:
        delegate_variable = Send(communicator, peer_rank=rank, peer_tag=tag)(x)

    delegate_variable.name = 'delegate_variable'
    return delegate_variable


def recv(communicator, rank, delegate_variable=None, tag=0, device=-1):
    """Receive elements from target process.

    This function returns data received from target process. If ``backward()``
    is invoked, it will try to send gradients to the target process.

    .. note::
        If you define non-connected computational graph on one process,
        you have to use ``delegate_variable`` to specify the output of
        previous computational graph component.
        Otherwise ``backward()`` does not work well.
        Please refer ``chainermn.functions.pseudo_connect`` for detail.

    Args:
        communicator (chainer.communicators.CommunicatorBase):
            ChainerMN communicator.
        rank (int): Target process specifier.
        delegate_variable (chainer.Variable):
            Pointer to the other non-connected component.
        tag (int): Optional message ID (MPI feature).
        device (int): Target device specifier.

    Returns:
        ~chainer.Variable:
            Data received from target process. If ``backward()`` is invoked
            by this variable, it will send gradients to the target process.

    """
    chainer.utils.experimental('chainermn.functions.recv')

    if rank == communicator.rank:
        raise ValueError(
            'rank must be different from communicator rank, '
            'otherwise deadlock occurs')

    if delegate_variable is None:
        return Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag,
            device=device)()
    else:
        delegate_variable.name = 'delegate_variable'
        return Recv(
            communicator,
            peer_rank=rank,
            peer_tag=tag,
            device=device)(delegate_variable)
