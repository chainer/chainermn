cdef extern from "cuda_runtime_api.h":
    ctypedef void* Stream 'struct CUstream_st*'


cdef extern from "nccl.h":
    ctypedef struct ncclComm:
        pass

    ctypedef ncclComm* ncclComm_t

    cdef enum:
        NCCL_UNIQUE_ID_BYTES = 128

    ctypedef struct ncclUniqueId:
        char internal[NCCL_UNIQUE_ID_BYTES]

    ctypedef enum ncclResult_t:
        ncclSuccess

    ctypedef enum ncclRedOp_t:
        pass

    ctypedef enum ncclDataType_t:
        pass

    const char* ncclGetErrorString(ncclResult_t result)

    ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)

    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev,
                                  ncclUniqueId commId, int rank)

    void ncclCommDestroy(ncclComm_t comm)

    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device)

    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank)

    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, Stream stream) nogil

    ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op, int root,
                            ncclComm_t comm, Stream stream) nogil

    ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                           int root, ncclComm_t comm, Stream stream) nogil

    ncclResult_t ncclReduceScatter(const void* sendbuff,
                                   void* recvbuff, int recvcount,
                                   ncclDataType_t datatype, ncclRedOp_t op,
                                   ncclComm_t comm, Stream stream) nogil

    ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
 +                             size_t count, ncclDataType_t datatype,
                               ncclComm_t comm, Stream stream) nogil

cdef dict STATUS_v1 = {
     0: 'NCCL_ERROR_SUCCESS',
     1: 'NCCL_ERROR_UNHANDLED_CUDA_ERROR',
     2: 'NCCL_ERROR_SYSTEM_ERROR',
     3: 'NCCL_ERROR_INTERNAL_ERROR',
     4: 'NCCL_ERROR_INVALID_DEVICE_POINTER',
     5: 'NCCL_ERROR_INVALID_RANK',
     6: 'NCCL_ERROR_UNSUPPORTED_DEVICE_COUNT',
     7: 'NCCL_ERROR_DEVICE_NOT_FOUND',
     8: 'NCCL_ERROR_INVALID_DEVICE_INDEX',
     9: 'NCCL_ERROR_LIB_WRAPPER_NOT_SET',
     10: 'NCCL_ERROR_CUDA_MALLOC_FAILED',
     11: 'NCCL_ERROR_RANK_MISMATCH',
     12: 'NCCL_ERROR_INVALID_ARGUMENT',
     13: 'NCCL_ERROR_INVALID_TYPE',
     14: 'NCCL_ERROR_INVALID_OPERATION',
}

cdef dict STATUS = {
     0: 'NCCL_ERROR_SUCCESS',
     1: 'NCCL_ERROR_UNHANDLED_CUDA_ERROR',
     2: 'NCCL_ERROR_SYSTEM_ERROR',
     3: 'NCCL_ERROR_INTERNAL_ERROR',
     4: 'NCCL_ERROR_INVALID_ARGUMENT',
     5: 'NCCL_ERROR_INVALID_USAGE',
}


cdef dict TYPE2TYPE_v1 = {
     NCCL_INT8: NCCL_CHAR_v1,
     NCCL_CHAR: NCCL_CHAR_v1,
     NCCL_UINT8: NCCL_INVALID_TYPE_v1,
     NCCL_INT32: NCCL_INT_v1,
     NCCL_INT: NCCL_INT_v1,
     NCCL_UINT32: NCCL_INVALID_TYPE_v1,
     NCCL_INT64: NCCL_INT64_v1,
     NCCL_UINT64: NCCL_UINT64_v1,
     NCCL_FLOAT16: NCCL_HALF_v1,
     NCCL_HALF: NCCL_HALF_v1,
     NCCL_FLOAT32: NCCL_FLOAT_v1,
     NCCL_FLOAT: NCCL_FLOAT_v1,
     NCCL_FLOAT64: NCCL_DOUBLE_v1
     NCCL_DOUBLE: NCCL_DOUBLE_v1,
}


class NcclError(RuntimeError):
    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        if NCCL_VERSION >= 2000:
            s = STATUS[status]
        else:
            s = STATUS_v1[status]
        super(Nccl1Error, self).__init__('%s: %s' % (s, msg.decode()))


cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        raise NcclError(status)


class NcclCommunicatorId(object):

    def __init__(self):
        cdef ncclUniqueId uniqueId
        status = ncclGetUniqueId(&uniqueId)
        check_status(status)
        self.data = []
        for i in range(NCCL_UNIQUE_ID_BYTES):
            self.data.append(<char>uniqueId.internal[i])


cdef struct comm_info:
    size_t ptr


class NcclCommunicator(object):

    def __init__(self, int ndev, comm_id, int rank):
        cdef ncclUniqueId _uniqueId
        for i in range(NCCL_UNIQUE_ID_BYTES):
            _uniqueId.internal[i] = comm_id.data[i]
        cdef ncclComm_t _comm
        status = ncclCommInitRank(&_comm, ndev, _uniqueId, rank)
        check_status(status)
        cdef comm_info _ci
        _ci.ptr = <size_t>_comm
        self.ci = _ci

    def destroy(self):
        cdef comm_info _ci = self.ci
        ncclCommDestroy(<ncclComm_t>_ci.ptr)

    def device_id(self):
        cdef comm_info _ci = self.ci
        cdef int device_id
        status = ncclCommCuDevice(<ncclComm_t>_ci.ptr, &device_id)
        check_status(status)
        return device_id

    def rank_id(self):
        cdef comm_info _ci = self.ci
        cdef int rank_id
        status = ncclCommUserRank(<ncclComm_t>_ci.ptr, &rank_id)
        check_status(status)
        return rank_id

    def allreduce(self, size_t sendbuf, size_t recvbuf,
                  int count, int datatype, int op, size_t stream):
        cdef comm_info _ci = self.ci
        with nogil:
             if NCCL_VERSION >= 2000:
                  status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, count,
                                    <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                    self._comm, <Stream>stream)
             else:
                  datatype_v1 = TYPE2TYPE_v1[datatype]
                  status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, <int>count,
                                    <ncclDataType_t>datatype_v1, <ncclRedOp_t>op,
                                    self._comm, <Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               int count, int datatype, int op, int root, size_t stream):
        cdef comm_info _ci = self.ci
        with nogil:
             if NCCL_VERSION >= 2000:
                  status = ncclReduce(<void*> sendbuf, <void*> recvbuf, count,
                                 <ncclDataType_t> datatype, <ncclRedOp_t> op,
                                 root, self._comm, <Stream> stream)
             else:
                  datatype_v1 = TYPE2TYPE_v1[datatype]
                  status = ncclReduce(<void*> sendbuf, <void*> recvbuf, <int> count,
                                 <ncclDataType_t> datatype_v1, <ncclRedOp_t> op,
                                 root, self._comm, <Stream> stream)
        check_status(status)

    def bcast(self, size_t buf, int count, int datatype, int root,
              size_t stream):
        cdef comm_info _ci = self.ci
        with nogil:
             if NCCL_VERSION >= 2000:
                  status = ncclBcast(<void*> buff, count,
                                <ncclDataType_t> datatype, root,
                                self._comm, <driver.Stream> stream)
             else:
                  datatype_v1 = TYPE2TYPE_v1[datatype]
                  status = ncclBcast(<void*> buff, <int> count,
                                <ncclDataType_t> datatype_v1, root,
                                self._comm, <driver.Stream> stream)
        check_status(status)

    def reduce_scatter(self, size_t sendbuf, size_t recvbuf,
                       int recvcount, int datatype, int op, size_t stream):
        cdef comm_info _ci = self.ci
        with nogil:
            if NCCL_VERSION >= 2000:
                status = ncclReduceScatter(
                    <void*>sendbuf, <void*>recvbuf, recvcount,
                    <ncclDataType_t>datatype, <ncclRedOp_t>op,
                    <ncclComm_t>_ci.ptr, <Stream>stream)
            else :
                datatype_v1 = TYPE2TYPE_v1[datatype]
                status = ncclReduceScatter(
                    <void*>sendbuf, <void*>recvbuf, recvcount,
                    <ncclDataType_t>datatype_v1, <ncclRedOp_t>op,
                    <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def allgather(self, size_t sendbuf, int count, int datatype,
                  size_t recvbuf, size_t stream):
        cdef comm_info _ci = self.ci
        with nogil:
            if NCCL_VERSION >= 2000:
                status = ncclAllGather(
                    <void*>sendbuf, <void*>recvbuf, count, <ncclDataType_t>datatype,
                    <ncclComm_t>_ci.ptr, <Stream>stream)
            else:
                datatype_v1 = TYPE2TYPE_v1[datatype]
                status = ncclAllGather(
                    <void*>sendbuf, <int> count, <ncclDataType_t>datatype,
                    <void*>recvbuf, <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)
