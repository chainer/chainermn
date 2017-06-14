import chainer.cuda
import math

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


class NonCudaAwareCommunicator(_base.NodeAwareCommunicatorBase):

    def __init__(self, mpi_comm):
        super(NonCudaAwareCommunicator, self).__init__(mpi_comm, use_nccl=True)
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def broadcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            data_xp = param.data
            data_np = chainer.cuda.to_cpu(data_xp)
            self.mpi_comm.Bcast(data_np)
            data_xp[:] = data_np

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            data_xp = param.data
            data_np = chainer.cuda.to_cpu(data_xp)
            self.mpi_comm.Allreduce(data_np)
            data_xp[:] = data_np
