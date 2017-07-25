import nose.plugins.skip
import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
import chainermn.functions
import copy
import numpy


@chainer.testing.parameterize(
    {'gpu': True},
    {'gpu': False},
)
class TestPointToPointCommunication(unittest.TestCase):

    def setUp(self):
        if self.gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            device = self.communicator.intra_rank
            chainer.cuda.get_device(device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            device = -1

        if self.communicator.size < 2:
            raise nose.plugins.skip.SkipTest()

        self.rank_send = (self.communicator.rank + 1) % self.communicator.size
        self.rank_recv = (self.communicator.rank - 1) % self.communicator.size

        # Activation function.
        self.f = chainer.functions.sigmoid

        # Evaluation function.
        self.evaluation = chainer.functions.mean_squared_error

        # Input data.
        self.x = chainer.Variable(
            numpy.arange(10).reshape(1, 10).astype(numpy.float32) / 10)

        self.model = chainer.links.Linear(
            10, 10, initialW=self._init_w(self.communicator.rank))
        self.entire_model = [chainer.links.Linear(
            10, 10, initialW=self._init_w(l))
            for l in range(self.communicator.size)]
        self.device = device

        if device >= 0:
            self.x.to_gpu()
            self.model.to_gpu()
            for model in self.entire_model:
                model.to_gpu()

    def _init_w(self, l):
        return 1.0 * numpy.arange(100).reshape(10, 10).astype(numpy.float32) \
            / ((l + 1) * 100)

    def test_communication(self):
        if self.communicator.rank == 0:
            # Input process.
            y = self.f(self.model(self.x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()
            grad = self.model.W.grad

            # Compute the expected gradient.
            x_ = self.x
            for l in range(self.communicator.size):
                x_ = self.f(self.entire_model[l](x_))
            err_ = self.evaluation(x_, self.x)
            err_.backward()
            grad_expected = self.entire_model[0].W.grad

            chainer.testing.assert_allclose(grad, grad_expected)

        elif self.communicator.rank == self.communicator.size - 1:
            # Output process.
            x = chainermn.functions.recv(
                self.communicator, self.rank_recv, device=self.device)
            y = self.f(self.model(x))
            err = self.evaluation(y, self.x)
            err.backward()

            # Compute the expected output.
            x_ = self.x
            for l in range(self.communicator.size):
                x_ = self.f(self.entire_model[l](x_))
            y_expect = x_

            chainer.testing.assert_allclose(y.data, y_expect.data)

        else:
            # Intermediate processes.
            x = chainermn.functions.recv(
                self.communicator, self.rank_recv, device=self.device)
            y = self.f(self.model(x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()

    def test_retain(self):
        if self.communicator.rank == 0:
            # Starting process.
            t = copy.copy(self.x)
            y = self.f(self.model(self.x))
            ptr = chainermn.functions.send(
                y, self.communicator, self.rank_send)

            # Unless recv_retain, backprop would stop here.
            x = chainermn.functions.recv(
                self.communicator, self.rank_recv,
                backward_pointer=ptr,
                device=self.device)
            err = self.evaluation(x, t)
            err.backward()

            # self.x.grad is None if backprop stops in the middle.
            self.assertIsNotNone(self.x.grad)

        else:
            # Intermediate processes.
            x = chainermn.functions.recv(
                self.communicator, self.rank_recv, device=self.device)
            y = self.f(self.model(x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()
