try:
    from chainermn.nccl.nccl import get_unique_id  # NOQA
    from chainermn.nccl.nccl import get_version  # NOQA
    from chainermn.nccl.nccl import NCCL_FLOAT  # NOQA
    from chainermn.nccl.nccl import NCCL_SUM  # NOQA
    from chainermn.nccl.nccl import NcclCommunicator  # NOQA
    from chainermn.nccl.nccl import NcclError  # NOQA
    _available = True
except ImportError:
    _available = False
