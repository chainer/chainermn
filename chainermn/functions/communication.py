import chainer
from chainer import cuda


class Send(chainer.Function):
    def __init__(self, comm, peer_rank, peer_tag, device=-1):
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag
        self.device = device

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        self.comm.send(x, self.peer_rank, self.peer_tag)
        return xp.array([]),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        with cuda.get_device(*inputs):
            gy = self.comm.recv(self.peer_rank, self.peer_tag)
            return xp.array(gy),


class Recv(chainer.Function):
    def __init__(self, comm, peer_rank, peer_tag, device=-1):
        self.comm = comm
        self.peer_rank = peer_rank
        self.peer_tag = peer_tag
        self.device = device

    def forward(self, inputs):
        x = self.comm.recv(self.peer_rank, self.peer_tag)
        if isinstance(self.device, int) and self.device >= 0:
            return cuda.to_gpu(x, device=self.device),
        else:
            return x,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*grad_outputs)
        gw, = grad_outputs
        self.comm.send(gw, self.peer_rank, self.peer_tag)
        return xp.array([])
