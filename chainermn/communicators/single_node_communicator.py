import chainer.cuda

from chainermn.communicators import _base
from chainermn.communicators import _memory_utility
from chainermn import nccl


class SingleNodeCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm):
        super(SingleNodeCommunicator, self).__init__(mpi_comm, use_nccl=True)

        if self.inter_size != 1:
            raise ValueError('SingleNodeCommunicator cannot be used under '
                             'multi-node settings')

        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

    def broadcast_data(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = [param for _, param in sorted(model.namedparams())]
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, itemsize, 'data', self.gpu_buffer_a)

        self.intra_nccl_comm.bcast(
            self.gpu_buffer_a.ptr(), n_elems_total, nccl.NCCL_FLOAT,
            0, stream.ptr)

        _memory_utility.unpack_params(
            params, itemsize, 'data', self.gpu_buffer_a)

    def allreduce_grad(self, model):
        self._init_comms()
        stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params(model)
        itemsize = 4
        n_elems_total = sum(param.grad.size for param in params)
        n_bytes_total = n_elems_total * itemsize
        self.gpu_buffer_a.assign(n_bytes_total)
        self.gpu_buffer_b.assign(n_bytes_total)

        _memory_utility.pack_params(
            params, itemsize, 'grad', self.gpu_buffer_a)

        self.intra_nccl_comm.allReduce(
            self.gpu_buffer_a.ptr(), self.gpu_buffer_b.ptr(), n_elems_total,
            nccl.NCCL_FLOAT, nccl.NCCL_SUM, stream.ptr)

        arr = self.gpu_buffer_b.array(n_elems_total)
        arr *= (1.0 / self.size)

        _memory_utility.unpack_params(
            params, itemsize, 'grad', self.gpu_buffer_b)
