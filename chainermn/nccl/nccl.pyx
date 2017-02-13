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

    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, Stream stream)

    ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, int count,
                            ncclDataType_t datatype, ncclRedOp_t op, int root,
                            ncclComm_t comm, Stream stream)

    ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype,
                           int root, ncclComm_t comm, Stream stream)


cdef dict STATUS = {
    0: 'NCCL_STATUS_SUCCESS',
    1: 'NCCL_STATUS_UNHANDLED_CUDA_ERROR',
    2: 'NCCL_STATUS_SYSTEM_ERROR',
    3: 'NCCL_STATUS_INTERNAL_ERROR',
    4: 'NCCL_STATUS_INVALID_DEVICE_POINTER',
    5: 'NCCL_STATUS_INVALID_RANK',
    6: 'NCCL_STATUS_UNSUPPORTED_DEVICE_COUNT',
    7: 'NCCL_STATUS_DEVICE_NOT_FOUND',
    8: 'NCCL_STATUS_INVALID_DEVICE_INDEX',
    9: 'NCCL_STATUS_LIB_WRAPPER_NOT_SET',
    10: 'NCCL_STATUS_CUDA_MALLOC_FAILED',
    11: 'NCCL_STATUS_RANK_MISMATCH',
    12: 'NCCL_STATUS_INVALID_ARGUMENT',
    13: 'NCCL_STATUS_INVALID_TYPE',
    14: 'NCCL_STATUS_INVALID_OPERATION',
}


cpdef enum:
    NCCL_SUM = 0
    NCCL_PROD = 1
    NCCL_MAX = 2
    NCCL_MIN = 3

    NCCL_CHAR = 0
    NCCL_INT = 1
    NCCL_HALF = 2
    NCCL_FLOAT = 3
    NCCL_DOUBLE = 4
    NCCL_INT64 = 5
    NCCL_UINT64 = 6


class NcclError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        msg = ncclGetErrorString(<ncclResult_t>status)
        super(NcclError, self).__init__('%s: %s' % (STATUS[status], msg))


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
        status = ncclAllReduce(<void*>sendbuf, <void*>recvbuf, count,
                               <ncclDataType_t>datatype, <ncclRedOp_t>op,
                               <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def reduce(self, size_t sendbuf, size_t recvbuf,
               int count, int datatype, int op, int root, size_t stream):
        cdef comm_info _ci = self.ci
        status = ncclReduce(<void*>sendbuf, <void*>recvbuf, count,
                            <ncclDataType_t>datatype, <ncclRedOp_t>op, root,
                            <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)

    def bcast(self, size_t buf, int count, int datatype, int root,
              size_t stream):
        cdef comm_info _ci = self.ci
        status = ncclBcast(<void*>buf, count, <ncclDataType_t>datatype, root,
                           <ncclComm_t>_ci.ptr, <Stream>stream)
        check_status(status)
