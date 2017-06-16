import chainer
from chainer import cuda
import chainer.utils


class Send(chainer.Function):
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
            # This dummy variable is necessary to backprop correctly in Chainer v2.
            # This trick relies on the fact that chainer.Variable.requires_grad is
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
