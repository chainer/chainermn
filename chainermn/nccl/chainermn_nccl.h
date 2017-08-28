#ifndef INCLUDE_GUARD_CHAINERMN_NCCL_H
#define INCLUDE_GUARD_CHAINERMN_NCCL_H

#include <nccl.h>

#ifndef NCCL_MAJOR
#define NCCL_MAJOR 1
#define NCCL_MINOR 0
#define NCCL_PATCH 0
#endif // #ifndef NCCL_MAJOR

#define NCCL_VERSION  (NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH)

#if (NCCL_VERSION < 2000)

#define NCCL_CHAR_V1 ncclChar
#define NCCL_INT_V1 ncclInt
#define NCCL_HALF_V1 ncclHalf
#define NCCL_FLOAT_V1 ncclFloat
#define NCCL_DOUBLE_v1 ncclDouble
#define NCCL_INT64_v1 ncclInt64
#define NCCL_UINT64_v1 ncclUint64
#define NCCL_INVALID_TYPE_V1 nccl_NUM_TYPES

static const ncclDataType_t TYPE2TYPE_V1[] = {
    NCCL_CHAR_V1,         // ncclInt8, ncclChar
    NCCL_INVALID_TYPE_V1, // ncclUint8
    NCCL_INT_V1,          // ncclInt32, ncclInt
    NCCL_INVALID_TYPE_V1, // ncclUint32
    NCCL_INT64_v1,        // ncclInt64
    NCCL_UINT64_v1,       // ncclUint64
    NCCL_HALF_V1,         // ncclFloat16, ncclHalf
    NCCL_FLOAT_V1,        // ncclFloat32, ncclFloat
    NCCL_DOUBLE_v1        // ncclFloat64, ncclDouble
};

#endif // #if (NCCL_VERSION < 2000)


ncclResult_t _ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
			    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
#if (NCCL_VERSION >= 2000)
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
#else 
    ncclDataType_t datatype_v1 = TYPE2TYPE_V1[datatype];
    return ncclAllReduce(sendbuff, recvbuff, count, datatype_v1, op, comm, stream);
#endif // #if (NCCL_VERSION >= 2000)
}


ncclResult_t _ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
			 ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
#if (NCCL_VERSION >= 2000)
    return ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
#else
    ncclDataType_t datatype_v1 = TYPE2TYPE_V1[datatype];
    return ncclReduce(sendbuff, recvbuff, count, datatype_v1, op, root, comm, stream);
#endif // #if (NCCL_VERSION >= 2000)
}


ncclResult_t _ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
		       ncclComm_t comm, cudaStream_t stream) {
#if (NCCL_VERSION >= 2000)
    return ncclBcast(buff, count, datatype, root, comm,  stream);
#else
    ncclDataType_t datatype_v1 = TYPE2TYPE_V1[datatype];
    return ncclBcast(buff, count, datatype_v1, root, comm,  stream);
#endif // #if (NCCL_VERSION >= 2000)
}


ncclResult_t _ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream) {
#if (NCCL_VERSION >= 2000)
    return ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
#else
    ncclDataType_t datatype_v1 = TYPE2TYPE_V1[datatype];
    return ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype_v1, op, comm, stream);
#endif // #if (NCCL_VERSION >= 2000)
}


ncclResult_t _ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
#if (NCCL_VERSION >= 2000)
    return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
#else
    ncclDataType_t datatype_v1 = TYPE2TYPE_V1[datatype];
    return ncclAllGather(sendbuff, sendcount, datatype_v1, recvbuff, comm, stream);
#endif // #if (NCCL_VERSION >= 2000)
}


#endif // #ifndef INCLUDE_GUARD_CHAINERMN_NCCL_H
