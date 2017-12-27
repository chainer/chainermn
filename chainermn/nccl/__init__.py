try:
    from cupy.cuda.nccl import get_unique_id  # NOQA
    from cupy.cuda.nccl import get_version  # NOQA
    from cupy.cuda.nccl import NCCL_FLOAT  # NOQA
    from cupy.cuda.nccl import NCCL_SUM  # NOQA
    from cupy.cuda.nccl import NcclCommunicator  # NOQA
    from cupy.cuda.nccl import NcclError  # NOQA
    _available = True
except ImportError:
    _available = False
