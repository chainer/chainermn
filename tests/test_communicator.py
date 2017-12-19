import mpi4py.MPI
import numpy as np
import pytest

import chainer
import chainer.cuda
import chainer.links
import chainer.testing
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


class ExampleModel(chainer.Chain):

    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )


class Param(object):
    def __init__(self, param):
        self.gpu = False
        self.nccl1 = False
        self.__dict__.update(param)


cpu_params = [Param(p) for p in [
    {
        'communicator_class': NaiveCommunicator,
        'multi_node': True,
    }]]
gpu_params = [Param(p) for p in [
    {
        'communicator_class': NaiveCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': HierarchicalCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': TwoDimensionalCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': SingleNodeCommunicator,
        'multi_node': False,
    }, {
        'communicator_class': NonCudaAwareCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
    }]]

mpi_comm = mpi4py.MPI.COMM_WORLD


def create_communicator(param, use_gpu):
    if not param.multi_node:
        ranks = _communication_utility.init_ranks(mpi_comm)
        inter_size = ranks[4]
        if inter_size > 1:
            pytest.skip('This test is for single node only')

    if use_gpu and not param.nccl1 and nccl.get_version() < 2000:
        pytest.skip('This test requires NCCL version >= 2.0')

    communicator = param.communicator_class(mpi_comm)

    if hasattr(communicator, 'intra_rank'):
        chainer.cuda.get_device(communicator.intra_rank).use()

    return communicator


def check_send_and_recv(communicator, *shape):
    if communicator.size < 2:
        pytest.skip("This test is for multiple nodes")

    if communicator.rank > 0:
        rank_prev = (communicator.rank - 1) % communicator.size
        data_recv = communicator.recv(source=rank_prev, tag=0)
        chainer.testing.assert_allclose(
            data_recv, rank_prev * np.ones((shape)))

    if communicator.rank < communicator.size - 1:
        rank_next = (communicator.rank + 1) % communicator.size
        data_send = communicator.rank * \
            np.ones((shape)).astype(np.float32)
        communicator.send(data_send, dest=rank_next, tag=0)


def check_send_and_recv_tuple(communicator, data):
    if communicator.size < 2:
        pytest.skip("This test is for multiple nodes")

    if communicator.rank > 0:
        rank_prev = (communicator.rank - 1) % communicator.size
        data_recv = communicator.recv(source=rank_prev, tag=0)
        for array0, array1 in zip(data, data_recv):
            chainer.testing.assert_allclose(array0, array1)

    if communicator.rank < communicator.size - 1:
        rank_next = (communicator.rank + 1) % communicator.size
        communicator.send(data, dest=rank_next, tag=0)


def check_broadcast_data(communicator, model):
    model.a.W.data[:] = communicator.rank
    model.b.W.data[:] = communicator.rank + 1
    model.c.b.data[:] = communicator.rank + 2
    communicator.broadcast_data(model)
    chainer.testing.assert_allclose(model.a.W.data, 0 * np.ones((3, 2)))
    chainer.testing.assert_allclose(model.b.W.data, 1 * np.ones((4, 3)))
    chainer.testing.assert_allclose(model.c.b.data, 2 * np.ones((5, )))


def check_allreduce_grad(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.
    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad[:] = communicator.rank + 2

        communicator.allreduce_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * np.ones((5, )))


def check_allreduce_grad_empty(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.
    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad = None

        communicator.allreduce_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))


def check_send_recv(param, use_gpu):
    communicator = create_communicator(param, use_gpu)

    assert mpi_comm.Get_rank() == communicator.rank
    assert mpi_comm.Get_size() == communicator.size

    check_send_and_recv(communicator, 50)
    check_send_and_recv(communicator, 50, 20)

    check_send_and_recv(communicator, 50, 20, 5)
    check_send_and_recv(communicator, 50, 20, 5, 3)

    data = [np.ones((50)).astype(np.float32)]
    check_send_and_recv_tuple(communicator, data)

    data = [
        np.ones((50)).astype(np.float32),
        np.ones((50, 20)).astype(np.float32),
        np.ones((50, 20, 5)).astype(np.float32)]
    check_send_and_recv_tuple(communicator, data)


def check_collective_communication(param, use_gpu):
    communicator = create_communicator(param, use_gpu)

    model = ExampleModel()
    if use_gpu:
        model.to_gpu()
    check_broadcast_data(communicator, model)
    check_allreduce_grad(communicator, model)
    check_allreduce_grad_empty(communicator, model)


# chainer.testing.parameterize is not available at functions
@pytest.mark.parametrize('param', cpu_params)
def test_communicator_cpu(param):
    check_send_recv(param, False)
    check_collective_communication(param, False)


@pytest.mark.parametrize('param', gpu_params)
@chainer.testing.attr.gpu
def test_communicator_gpu(param):
    check_send_recv(param, True)
    check_collective_communication(param, True)
