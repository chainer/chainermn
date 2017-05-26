#!/usr/bin/env python
# coding: utf-8

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

import chainermn
import chainermn.functions


class MLP0(chainer.Chain):
    def __init__(self, comm, n_out):
        super(MLP0, self).__init__(
            l1=L.Linear(784, n_out),
        )
        self.comm = comm

    def __call__(self, x, t_dummy):
        # the argument `t_dummy` is necessary for receiving TupleDataset
        h1 = F.relu(self.l1(x))
        return chainermn.functions.Send(
            self.comm, peer_rank=1, peer_tag=0, device=self._device_id)(h1)


class MLP1(chainer.Chain):
    def __init__(self, comm, n_units, n_out):
        super(MLP1, self).__init__(
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out),
        )
        self.comm = comm

    def __call__(self, x_dummy):
        # the argument `x_dummy` is necessary for receiving TupleDataset
        h1 = chainermn.functions.Recv(
            self.comm, peer_rank=0, peer_tag=0, device=self._device_id)()
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ChainerMN example: pipelined neural network')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        device = -1

    print('GPU: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    if comm.rank == 0:
        # The former half of the model is placed in rank=0.
        model = MLP0(comm, args.unit)
    elif comm.rank == 1:
        # The latter half of the model is placed in rank=1.
        model = L.Classifier(MLP1(comm, args.unit, 10))

    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Random seed must be fixed to distribute the same dataset to all machines.
    # (This line must be executed after model initializations, somehow)
    np.random.seed(0)

    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    if comm.rank == 1:
        # Report on rank=1.
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    trainer.run()
