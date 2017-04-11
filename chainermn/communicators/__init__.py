import mpi4py.MPI


def create_communicator(communicator_name='hierarchical', mpi_comm=mpi4py.MPI.COMM_WORLD):
    """Create a ChainerMN communicator.

    Different communicators provide different approaches of communication, so
    they have different performance charasteristics. The default communicator
    ``hierarchical`` is expected to generally perform well on a variety of
    environments, so one need not to change communicators in most cases.
    However, choosing proper communicator may give better performance.
    The following communicators are available.

    =============== === === ======== =======================================
    Name            CPU GPU NCCL     Recommended Use Cases
    =============== === === ======== =======================================
    naive           OK  OK           Testing on CPU mode
    flat                OK           Each node has one GPU
    hierarchical        OK  Required Each node has multiple GPUs and one HCA
    two_dimensional     OK  Required Each node has multiple GPUs and HCAs
    single_node         OK  Required Single node with multiple GPUs
    =============== === === ======== =======================================

    Args:
        communicator_name: The name of communicator (``naive``, ``flat``,
          ``hierarchical``, ``two_dimensional``, or ``single_node``)
        mpi_comm: MPI4py communicator

    Returns:
        ChainerMN communicator


    """
    if communicator_name == 'naive':
        from chainermn.communicators.naive_communicator \
            import NaiveCommunicator
        return NaiveCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'flat':
        from chainermn.communicators.flat_communicator \
            import FlatCommunicator
        return FlatCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'hierarchical':
        from chainermn.communicators.hierarchical_communicator \
            import HierarchicalCommunicator
        return HierarchicalCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'two_dimensional':
        from chainermn.communicators.two_dimensional_communicator \
            import TwoDimensionalCommunicator
        return TwoDimensionalCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'single_node':
        from chainermn.communicators.single_node_communicator \
            import SingleNodeCommunicator
        return SingleNodeCommunicator(mpi_comm=mpi_comm)

    elif communicator_name == 'dummy':
        from chainermn.communicators.dummy_communicator \
            import DummyCommunicator
        return DummyCommunicator(mpi_comm=mpi_comm)

    else:
        raise ValueError(
            'Unrecognized communicator: "{}"'.format(communicator_name))
