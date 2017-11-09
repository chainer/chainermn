def create_multi_node_optimizer(actual_optimizer, communicator,
                                double_buffering=False):
    """Create a multi node optimizer from a Chainer optimizer.

    Args:
        actual_optimizer: Chainer optimizer
            (e.g., ``chainer.optimizers.Adam``).
        communicator: ChainerMN communicator.
        double_buffering: double buffering flag. It is supported
            by PureNcclCommunicator only.
    Returns:
        The multi node optimizer based on ``actual_optimizer``.
    """
    if double_buffering:
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        if not isinstance(communicator, PureNcclCommunicator): 
            raise ValueError(
                'This communicator does not support double buffering.')
        from chainermn.optimizers.double_buffering_optimizer \
            import DoubleBufferingOptimizer
        return DoubleBufferingOptimizer(actual_optimizer, communicator)

    from chainermn.optimizers.multi_node_optimizer \
        import MultiNodeOptimizer
    return MultiNodeOptimizer(actual_optimizer, communicator)
