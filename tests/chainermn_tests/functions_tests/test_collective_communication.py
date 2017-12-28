import chainer
import numpy
import pytest

import chainermn
import chainermn.functions


def create_communicator_and_device(gpu):
    if gpu:
        communicator = chainermn.create_communicator('hierarchical')
        device = communicator.intra_rank
        chainer.cuda.get_device(device).use()
    else:
        communicator = chainermn.create_communicator('naive')
        device = -1

    if communicator.size < 2:
        pytest.skip("This test is for multinode")

    return communicator, device


def check_all_to_all(communicator, device, xs):
    ys = chainermn.functions.all_to_all(communicator, xs, device)

    y = chainer.functions.sum(ys[0])
    for _y in ys[1:]:
        y += chainer.functions.sum(_y)

    y.backward()

    assert xs[0].grad is not None


def test_all_to_all_cpu():
    communicator, device = create_communicator_and_device(False)
    data = [
        chainer.Variable(numpy.zeros(
            (communicator.rank, i), dtype=numpy.float32))
        for i in range(communicator.size)]
    check_all_to_all(communicator, device, data)


@chainer.testing.attr.gpu
def test_all_to_all_gpu():
    communicator, device = create_communicator_and_device(True)

    chainer.cuda.get_device_from_id(device).use()
    data = [
        chainer.Variable(numpy.zeros(
            (communicator.rank, i), dtype=numpy.float32))
        for i in range(communicator.size)]
    for x in data:
        x.to_gpu()
    check_all_to_all(communicator, device, data)
