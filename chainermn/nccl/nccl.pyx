cdef extern from "cuda_runtime_api.h":
    ctypedef void* Stream 'struct CUstream_st*'


cdef extern from "chainermn_nccl.h":
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

    ncclResult_t _ncclAllReduce(const void* sendbuff, void* recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm,
                                Stream stream) nogil

    ncclResult_t _ncclReduce(const void* sendbuff, void* recvbuff,
                             size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, int root, ncclComm_t comm,
                             Stream stream) nogil

    ncclResult_t _ncclBcast(void* buff, size_t count, ncclDataType_t datatype,
                            int root, ncclComm_t comm, Stream stream) nogil

    ncclResult_t _ncclReduceScatter(const void* sendbuff,
                                    void* recvbuff, size_t recvcount,
                                    ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, Stream stream) nogil

    ncclResult_t _ncclAllGather(const void* sendbuff, void* recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclComm_t comm, Stream stream) nogil
    int NCCL_VERSION

cdef dict STATUS_V1 = {
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


class NcclError(RuntimeError):
    def __init__(self, int status):
        self.status = status
        cdef msg = ncclGetErrorString(<ncclResult_t>status)
        if NCCL_VERSION >= 2000:
            s = STATUS[status]
        else:
            s = STATUS_V1[status]
        super(NcclError, self).__init__('%s: %s' % (s, msg.decode()))


cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        raise NcclError(status)


def get_version():
    return NCCL_VERSION


def get_unique_id():
    cdef ncclUniqueId uniqueId
    status = ncclGetUniqueId(&uniqueId)
    check_status(status)
    ret = tuple([<char>uniqueId.internal[i]
                 for i in range(NCCL_UNIQUE_ID_BYTES)])
    return ret


cdef struct CommInfo:
    size_t ptr


class NcclCommunicator(object):

    def __init__(self, int ndev, comm_id, int rank):
        cdef ncclUniqueId _uniqueId
        for i in range(NCCL_UNIQUE_ID_BYTES):
            _uniqueId.internal[i] = comm_id[i]
        cdef ncclComm_t _comm
        status = ncclCommInitRank(&_comm, ndev, _uniqueId, rank)
        check_status(status)
        cdef CommInfo _ci
        _ci.ptr = <size_t>_comm
        self.ci = _ci

    def destroy(self):
        cdef CommInfo _ci = self.ci
        ncclCommDestroy(<ncclComm_t>_ci.ptr)

    def device_id(self):
        cdef CommInfo _ci = self.ci
        cdef int device_id
        status = ncclCommCuDevice(<ncclComm_t>_ci.ptr, &device_id)
        check_status(status)
        return device_id

    def rank_id(self):
        cdef CommInfo _ci = self.ci
        cdef int rank_id
        status = ncclCommUserRank(<ncclComm_t>_ci.ptr, &rank_id)
        check_status(status)
        return rank_id

    def allreduce(self, size_t sendbuf, size_t recvbuf,
                  int count, int datatype, int op, size_t stream):
        cdef CommInfo _ci = self.ci
        with nogil:
            status = _ncclAllReduce(<void*>sendbuf, <void*>recvbuf, count,
                                    <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                    <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               int count, int datatype, int op, int root, size_t stream):
        cdef CommInfo _ci = self.ci
        with nogil:
            status = _ncclReduce(<void*> sendbuf, <void*> recvbuf, count,
                                 <ncclDataType_t> datatype, <ncclRedOp_t> op,
                                 root, <ncclComm_t>_ci.ptr, <Stream> stream)
        check_status(status)

    def bcast(self, size_t buf, int count, int datatype, int root,
              size_t stream):
        cdef CommInfo _ci = self.ci
        with nogil:
            status = _ncclBcast(
                <void*>buf, count, <ncclDataType_t>datatype, root,
                <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def reduce_scatter(self, size_t sendbuf, size_t recvbuf,
                       int recvcount, int datatype, int op, size_t stream):
        cdef CommInfo _ci = self.ci
        with nogil:
            status = _ncclReduceScatter(
                <void*>sendbuf, <void*>recvbuf, recvcount,
                <ncclDataType_t>datatype, <ncclRedOp_t>op,
                <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def allgather(self, size_t sendbuf, int count, int datatype,
                  size_t recvbuf, size_t stream):
        cdef CommInfo _ci = self.ci
        with nogil:
            status = _ncclAllGather(<void*>sendbuf, <void*>recvbuf,
                                    count, <ncclDataType_t>datatype,
                                    <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)
