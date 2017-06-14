import chainer.cuda
import math
import mpi4py

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
            data = param.data
            tmp_cpu = chainer.cuda.to_cpu(data)
            self.mpi_comm.Bcast(tmp_cpu)
            tmp_gpu = chainer.cuda.to_gpu(tmp_cpu)
            data[:] = tmp_gpu

    def allreduce_grad(self, model):
        for _, param in sorted(model.namedparams()):
            data = param.grad
            tmp_cpu = chainer.cuda.to_cpu(data)
            self.mpi_comm.Allreduce(mpi4py.MPI.IN_PLACE, tmp_cpu)
            tmp_gpu = chainer.cuda.to_gpu(tmp_cpu)
            data[:] = tmp_gpu
            data /= self.size
