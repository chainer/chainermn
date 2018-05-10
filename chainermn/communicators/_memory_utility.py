import ctypes

import cffi
import mpi4py.MPI
import numpy as np

import chainer.cuda
try:
    import cupy as cp
    _cupy_avail = True
except ImportError:
    _cupy_avail = False


class HostPinnedMemory(object):

    def __init__(self):
        if not _cupy_avail:
            raise RuntimeError("HostPinnedMemory cannot be used: " +
                               "Cupy is not available.")
        self.size = 0
        self.memory = None
        self.cptr = None
        self.ffi = cffi.FFI()

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc_pinned_memory(size)
            self.cptr = self.ffi.cast('void *', self.memory.ptr)

    def ptr(self, offset=0):
        return ctypes.c_void_p(self.memory.ptr + offset)

    def buffer(self, size):
        return self.ffi.buffer(self.cptr, size)

    def array(self, count, offset=0, dtype=np.float32):
        if dtype is None:
            raise TypeError('dtype must be an instance of numpy.dtype class')
        return np.frombuffer(
            self.memory, count=count, offset=offset, dtype=dtype)


class DeviceMemory(object):

    def __init__(self):
        if not _cupy_avail:
            raise RuntimeError("DeviceMemory cannot be used: " +
                               "Cupy is not available.")
        self.size = 0
        self.memory = None
        self.cptr = None
        self.ffi = cffi.FFI()

    def assign(self, size):
        if size > self.size:
            self.size = size
            self.memory = cp.cuda.alloc(size)

    def from_device(self, src, size, offset=0, stream=None):
        dst = self.memory + offset
        if stream is None:
            dst.copy_from_device(src.data, size)
        else:
            dst.copy_from_device_async(src.data, size, stream)

    def to_device(self, dst, size, offset=0, stream=None):
        src = self.memory + offset
        if stream is None:
            dst.data.copy_from_device(src, size)
        else:
            dst.data.copy_from_device_async(src, size, stream)

    def ptr(self):
        return self.memory.ptr

    def buffer(self, size):
        return self.ffi.buffer(self.ffi.cast('void *', self.memory.ptr), size)

    def array(self, shape, offset=0, dtype=np.float32):
        if dtype is None:
            raise TypeError('dtype must be an instance of numpy.dtype class')
        return cp.ndarray(shape, memptr=self.memory + offset, dtype=dtype)


def extract_params(model):
    return [param for _, param in sorted(model.namedparams())
            if param.grad is not None]


def pack_params(params, itemsize, attr_name, buffer, stream=None):
    offset = 0
    for param in params:
        grad = getattr(param, attr_name)
        size = grad.size * itemsize
        buffer.from_device(grad, size, offset, stream)
        offset += size


def unpack_params(params, itemsize, attr_name, buffer, stream=None):
    offset = 0
    for param in params:
        grad = getattr(param, attr_name)
        size = grad.size * itemsize
        buffer.to_device(grad, size, offset, stream)
        offset += size


def array_to_buffer_object(array, mpi_dtype=mpi4py.MPI.FLOAT):
    xp = chainer.cuda.get_array_module(array)

    if xp is np:
        return get_device_memory_pointer(array)
    else:
        return (get_device_memory_pointer(array), mpi_dtype)


def get_device_memory_pointer(array):
    xp = chainer.cuda.get_array_module(array)
    array = xp.ascontiguousarray(array)

    if xp is np:
        return array
    else:
        ffi = cffi.FFI()
        return ffi.buffer(ffi.cast('void *', array.data.ptr), array.nbytes)
