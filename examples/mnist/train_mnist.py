#!/usr/bin/env python
from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import chainermn


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(784, n_units),  # n_in -> n_units
            l2=L.Linear(n_units, n_units),  # n_units -> n_units
            l3=L.Linear(n_units, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    # TODO(akiba): write a comment about devices
    if args.gpu:
        comm = chainermn.NodeAwareCommunicator()
        device = comm.intra_rank
    else:
        comm = chainermn.NaiveCommunicator()
        device = -1

    print('GPU: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    model = L.Classifier(MLP(args.unit, 10))
    if device >= 0:
        chainer.cuda.get_device(device).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # TODO(akiba): write a comment about optimizers
    optimizer = chainermn.MultiNodeOptimizer(chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # TODO(akiba): write a comment about datasets
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None
    train = chainermn.scatter_dataset(train, comm)
    test = chainermn.scatter_dataset(test, comm)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, chainermn.get_epoch_trigger(
        args.epoch, train, args.batchsize, comm), out=args.out)

    # TODO(akiba): write a comment about evaluators
    trainer.extend(
        chainermn.MultiNodeEvaluator(comm, test_iter, model, device=device),
        trigger=chainermn.get_epoch_trigger(1, train, args.batchsize, comm))

    # TODO(akiba): write a comment about master-only extensions
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(['main/accuracy',
                                       'validation/main/accuracy'],
                                      'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
