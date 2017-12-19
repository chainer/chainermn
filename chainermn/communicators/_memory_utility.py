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

    def array(self, count, offset=0):
        return np.frombuffer(
            self.memory, count=count, offset=offset, dtype=cp.float32)


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

    def from_device(self, src, size, offset=0):
        dst = self.memory + offset
        dst.copy_from_device(src.data, size)

    def to_device(self, dst, size, offset=0):
        src = self.memory + offset
        dst.data.copy_from_device(src, size)

    def ptr(self):
        return self.memory.ptr

    def buffer(self, size):
        return self.ffi.buffer(self.ffi.cast('void *', self.memory.ptr), size)

    def array(self, shape, offset=0):
        return cp.ndarray(shape, memptr=self.memory + offset, dtype=cp.float32)


def extract_params(model):
    return [param for _, param in sorted(model.namedparams())
            if param.grad is not None]


def pack_params(params, itemsize, attr_name, buffer):
    offset = 0
    for param in params:
        grad = getattr(param, attr_name)
        size = grad.size * itemsize
        buffer.from_device(grad, size, offset)
        offset += size


def unpack_params(params, itemsize, attr_name, buffer):
    offset = 0
    for param in params:
        grad = getattr(param, attr_name)
        size = grad.size * itemsize
        buffer.to_device(grad, size, offset)
        offset += size


def array_to_buffer_object(array):
    if chainer.cuda.get_array_module(array) is np:
        return array
    else:
        ffi = cffi.FFI()
        return (ffi.buffer(ffi.cast('void *', array.data.ptr), array.nbytes),
                mpi4py.MPI.FLOAT)
