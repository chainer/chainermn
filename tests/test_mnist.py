# coding: utf-8

import sys
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainermn


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(784, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class TestMNIST(unittest.TestCase):
    def test_mnist(self, display_log=True):
        # This test file is intended to be run on Travis-CI and
        # GPU is not used for now.
        epoch = 10
        batchsize = 100
        n_units = 100

        comm = chainermn.create_communicator('naive')
        model = L.Classifier(MLP(n_units, 10))
        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), comm)
        optimizer.setup(model)

        if comm.rank == 0:
            train, test = chainer.datasets.get_mnist()
        else:
            train, test = None, None

        train = chainermn.scatter_dataset(train, comm)
        test = chainermn.scatter_dataset(test, comm)

        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                     repeat=False,
                                                     shuffle=False)

        updater = training.StandardUpdater(train_iter, optimizer)
        trainer = training.Trainer(updater, chainermn.get_epoch_trigger(
            epoch, train, batchsize, comm))

        # Wrap standard Chainer evaluators by MultiNodeEvaluator.
        evaluator = extensions.Evaluator(test_iter, model)
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
        trainer.extend(
            evaluator,
            trigger=chainermn.get_epoch_trigger(1, train, batchsize, comm))

        # Some display and output extensions are necessary only for one worker.
        # (Otherwise, there would just be repeated outputs.)
        if comm.rank == 0 and display_log:
            trainer.extend(extensions.LogReport(trigger=(1, 'epoch')),
                           trigger=(1, 'epoch'))
            trainer.extend(extensions.PrintReport(['epoch',
                                                   'main/loss',
                                                   'validation/main/loss',
                                                   'main/accuracy',
                                                   'validation/main/accuracy',
                                                   'elapsed_time'],
                                                  out=sys.stderr),
                           trigger=(1, 'epoch'))
        trainer.run()

        err = evaluator()['validation/main/accuracy']
        self.assertGreaterEqual(err, 0.95)


if __name__ == "__main__":
    TestMNIST().test_mnist(display_log=True)
