import mpi4py.MPI


def create_communicator(communicator_name='hierarchical', mpi_comm=mpi4py.MPI.COMM_WORLD):
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
