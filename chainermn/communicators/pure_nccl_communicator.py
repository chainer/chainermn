import chainer.cuda

from chainermn.communicators import _base
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl

import numpy as np

class PureNcclCommunicator(_base.CommunicatorBase):

    def __init__(self, mpi_comm, allreduce_grad_dtype=None):
        if nccl.get_version() < 2000:
            raise RuntimeError(
                'PureNcclCommunicator is only supported on NCCL 2.0+')

        super(PureNcclCommunicator, self).__init__(mpi_comm, True)
        self._init_ranks()

        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        self.intra_nccl_comm = None
        self.nccl_comm = None

        self.gpu_tmp_grad_buffer = _memory_utility.DeviceMemory()                
        self.gpu_allreduce_grad_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_allreduce_grad_buffer_b = _memory_utility.DeviceMemory()

        if allreduce_grad_dtype is not None:
            self.allreduce_grad_dtype = np.dtype(allreduce_grad_dtype)
            if self.allreduce_grad_dtype.kind != 'f':
                raise ValueError(
                    'allreduce_grad_dtype must be float16, float32, float64, or None.')
        else:
            self.allreduce_grad_dtype = None
        self.grad_dtype_to_allreduce_dtype_kernel = None
        self.allreduce_dtype_to_grad_dtype_kernel = None
        
    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        if self.inter_mpi_comm is not None:
            assert self.intra_mpi_comm is not None
            assert self.intra_nccl_comm is not None
            assert self.nccl_comm is not None
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=True)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        self.intra_nccl_comm = comms[2]
        self.nccl_comm = comms[3]

    def broadcast_data(self, model):
        _communication_utility.broadcast_naive(self.mpi_comm, model)

    def allreduce_grad(self, model, stream=None):
        self._init_comms()
        if stream is None:
            stream = chainer.cuda.Stream.null

        params = _memory_utility.extract_params(model)
        grad_dtype = _get_param_grad_dtype(params[0])
        if self.allreduce_grad_dtype is None:
            allreduce_grad_dtype = grad_dtype
        else :
            allreduce_grad_dtype = self.allreduce_grad_dtype
        grad_dtype_itemsize = grad_dtype.itemsize
        allreduce_grad_dtype_itemsize = allreduce_grad_dtype.itemsize 
        n_elems = sum(param.grad.size for param in params)
        grad_n_bytes = grad_dtype_itemsize * n_elems
        allreduce_grad_n_bytes = allreduce_grad_dtype_itemsize * n_elems
        
        self.gpu_allreduce_grad_buffer_a.assign(allreduce_grad_n_bytes)
        self.gpu_allreduce_grad_buffer_b.assign(allreduce_grad_n_bytes)
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.pack_params(
                params, allreduce_grad_dtype_itemsize, 'grad', 
                self.gpu_allreduce_grad_buffer_a)
        else:
            if self.grad_dtype_to_allreduce_dtype_kernel is None:
                self.grad_dtype_to_allreduce_dtype_kernel = _get_converting_kernel(
                    grad_dtype, allreduce_grad_dtype, 'grad_dtype_to_allreduce_dtype_kernel')
            self.gpu_tmp_grad_buffer.assign(grad_n_bytes)
            _memory_utility.pack_params(
                params, grad_dtype_itemsize, 'grad', 
                self.gpu_tmp_grad_buffer)
            self.grad_dtype_to_allreduce_dtype_kernel(
                self.gpu_tmp_grad_buffer.array(n_elems, dtype=grad_dtype),
                self.gpu_allreduce_grad_buffer_a.array(n_elems, dtype=allreduce_grad_dtype), 
                stream=stream)

        self.nccl_comm.allreduce(self.gpu_allreduce_grad_buffer_a.ptr(),
                                 self.gpu_allreduce_grad_buffer_b.ptr(), n_elems,
                                 _get_nccl_type_id(allreduce_grad_dtype), 
                                 nccl.NCCL_SUM,
                                 stream.ptr)
        if stream != chainer.cuda.Stream.null:
            stream.synchronize()
        if grad_dtype == allreduce_grad_dtype:
            ret = self.gpu_allreduce_grad_buffer_b.array(n_elems) * (1.0 / self.size)
            self.gpu_allreduce_grad_buffer_b.from_device(ret, allreduce_grad_n_bytes)
            _memory_utility.unpack_params(
                params, allreduce_grad_dtype_itemsize, 'grad', self.gpu_allreduce_grad_buffer_b)

        else:
            if self.allreduce_dtype_to_grad_dtype_kernel is None:
                self.allreduce_dtype_to_grad_dtype_kernel = _get_converting_kernel(
                    allreduce_grad_dtype, grad_dtype, 'allreduce_dtype_to_grad_dtype_kernel')
            self.allreduce_dtype_to_grad_dtype_kernel(
                self.gpu_allreduce_grad_buffer_b.array(n_elems, dtype=allreduce_grad_dtype),
                self.gpu_tmp_grad_buffer.array(n_elems, dtype=grad_dtype),
                stream=stream)
            ret = self.gpu_tmp_grad_buffer.array(n_elems) * (1.0 / self.size)
            self.gpu_tmp_grad_buffer.from_device(ret, grad_n_bytes)
            _memory_utility.unpack_params(
                params, grad_dtype_itemsize, 'grad', self.gpu_tmp_grad_buffer)

        
def _get_converting_kernel(src_dtype, dst_dtype, kernel_name):
    return chainer.cuda.cupy.ElementwiseKernel(
        '{} x'.format(src_dtype.name), '{} y'.format(dst_dtype.name), 'y = x', kernel_name)


def _get_param_grad_dtype(param):
    return param.grad.dtype


def _get_nccl_type_id(dtype):
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCC_FLOAT64
    else:
        raise ValueError(
            'dtype must be float16, float32, or float64.')
