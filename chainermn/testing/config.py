from chainermn.communicators import _communication_utility
from chainermn.communicators.flat_communicator \
    import FlatCommunicator
from chainermn.communicators.hierarchical_communicator \
    import HierarchicalCommunicator
from chainermn.communicators.naive_communicator \
    import NaiveCommunicator
from chainermn.communicators.non_cuda_aware_communicator \
    import NonCudaAwareCommunicator
from chainermn.communicators.pure_nccl_communicator \
    import PureNcclCommunicator
from chainermn.communicators.single_node_communicator \
    import SingleNodeCommunicator
from chainermn.communicators.two_dimensional_communicator \
    import TwoDimensionalCommunicator
from chainermn import nccl


def get_communicator(gpu=False):
    if gpu:
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank
        chainer.cuda.get_device(device).use()
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    return comm
