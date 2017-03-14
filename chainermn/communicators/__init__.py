def get_communicator(communicator_name='node_aware', *args, **kwargs):
    if communicator_name == 'naive':
        from chainermn.communicators.naive_communicator \
            import NaiveCommunicator
        return NaiveCommunicator(*args, **kwargs)
    elif communicator_name == 'node_aware':
        from chainermn.communicators.node_aware_communicator \
            import NodeAwareCommunicator
        return NodeAwareCommunicator(*args, **kwargs)
    else:
        raise ValueError(
            'Unrecognized communicator: "{}"'.format(communicator_name))
