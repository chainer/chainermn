# encoding: utf-8

import argparse
import collections
import os.path
import pickle
import re
import sys
import time

from nltk.corpus import comtrans
from nltk.translate import bleu_score
import numpy
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions
import chainermn

import europal


def cached_call(fname, func, *args):
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        # not yet cached
        val = func(*args)
        with open(fname, 'wb') as f:
            pickle.dump(val, f)
        return val


def read_source(in_dir, cache=None):
    en_path = os.path.join(in_dir, 'giga-fren.release2.fixed.en')
    source_vocab = ['<eos>', '<unk>'] + europal.count_words(en_path)
    source_data = europal.make_dataset(en_path, source_vocab)

    return source_vocab, source_data


def read_target(in_dir, cahce=None):
    fr_path = os.path.join(in_dir, 'giga-fren.release2.fixed.fr')
    target_vocab = ['<eos>', '<unk>'] + europal.count_words(fr_path)
    target_data = europal.make_dataset(fr_path, target_vocab)

    return target_vocab, target_data


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            decoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            W=L.Linear(n_units, n_target_vocab),
        )
        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, *inputs):
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]

        xs = [x[::-1] for x in xs]

        eos = self.xp.zeros(1, 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        reporter.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                xs = [x[::-1] for x in xs]
                # exs = sequence_embed(self.embed_x, [self.xp.array(x) for x in xs])
                exs = sequence_embed(self.embed_x, xs)
                # Initial hidden variable and cell variable
                # zero = self.xp.zeros((self.n_layers, batch, self.n_units), 'f')
                # h, c, _ = self.encoder(zero, zero, exs, train=False)
                h, c, _ = self.encoder(None, None, exs)
                ys = self.xp.zeros(batch, 'i')
                result = []
                for i in range(max_length):
                    eys = self.embed_y(ys)
                    eys = chainer.functions.split_axis(
                        eys, batch, 0, force_tuple=True)
                    h, c, ys = self.decoder(h, c, eys)
                    cys = chainer.functions.concat(ys, axis=0)
                    wy = self.W(cys)
                    ys = self.xp.argmax(wy.data, axis=1).astype('i')
                    result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return tuple(
        to_device_batch([x for x, _ in batch]) +
        to_device_batch([y for _, y in batch]))


class CalculateBleu(chainer.training.Extension):
    # priority = chainer.training.PRIORITY_WRITER
    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        reporter.report({self.key: bleu})


class BleuEvaluator(extensions.Evaluator):
    def __init__(self, model, test_data, device=-1, batch=100,
                 max_length=100, comm=None):
        super(BleuEvaluator, self).__init__({'main': None}, model)
        self.model = model
        self.test_data = test_data
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.comm = comm

    def evaluate(self):
        bt = time.time()
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            observation = {}
            with reporter.report_scope(observation):
                for i in range(0, len(self.test_data), self.batch):
                    src, trg = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in trg])

                    src = [chainer.dataset.to_device(self.device, x)
                           for x in src]
                    ys = [y.tolist()
                          for y in self.model.translate(src, self.max_length)]
                    hypotheses.extend(ys)

                bleu = bleu_score.corpus_bleu(
                    references, hypotheses,
                    smoothing_function=bleu_score.SmoothingFunction().method1)
                reporter.report({'bleu': bleu}, self.model)
        et = time.time()

        if self.comm is not None:
            # This evaluator is called via chainermn.MultiNodeEvaluator
            for i in range(0, self.comm.mpi_comm.size):
                print("BleuEvaluator::evaluate(): "
                      "took {:.3f} [s]".format(et - bt))
                sys.stdout.flush()
                self.comm.mpi_comm.Barrier()
        else:
            # This evaluator is called from a conventional
            # Chainer exntension
            print("BleuEvaluator(single)::evaluate(): "
                  "took {:.3f} [s]".format(et - bt))
            sys.stdout.flush()
        return observation


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch') # 10 - 13,14 epoch?
    parser.add_argument('--bleu', type=bool, default=False,
                        help='Report BLEU score')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--cache', '-c', default=None,
                        help='Directory to cache pre-processed dataset')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    parser.add_argument('--shift-gpu', type=int, default=0,
                        help=("Shift device ID of GPUs " +
                              "(convenient if you want to " +
                              "run multiple seq2seq_mn.py on a sinle node"))
    parser.add_argument('--stop', '-s', type=str, default="15e",
                        help='Stop trigger (ex. "500i", "15e")')
    parser.add_argument('--input', '-i', type=str, default='wmt',
                        help='Input directory')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    # Prepare ChainerMN communicator
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        dev = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        dev = -1

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    if False:
        sentences = comtrans.aligned_sents('alignment-en-fr.txt')
        source_ids = collections.defaultdict(lambda: len(source_ids))
        target_ids = collections.defaultdict(lambda: len(target_ids))
        target_ids['eos']
        data = []
        for sentence in sentences:
            source = numpy.array([source_ids[w] for w in sentence.words], 'i')
            target = numpy.array([target_ids[w] for w in sentence.mots], 'i')
            data.append((source, target))
        print('Source vocabulary: %d' % len(source_ids))
        print('Target vocabulary: %d' % len(target_ids))

        test_data = data[:len(data) / 10]
        train_data = data[len(data) / 10:]
    else:
        # Rank 0 prepares all data
        if comm.rank == 0:
            if args.cache and not os.path.exists(args.cache):
                os.mkdir(args.cache)

            # Read source data
            bt = time.time()
            if args.cache:
                cache_file = os.path.join(args.cache, 'source.pickle')
                source_vocab, source_data = cached_call(cache_file,
                                                        read_source,
                                                        args.input, args.cache)
            else:
                source_vocab, source_data = read_source(args.input, args.cache)
            et = time.time()
            print("RD source done. {:.3f} [s]".format(et - bt))
            sys.stdout.flush()

            # Read target data
            bt = time.time()
            if args.cache:
                cache_file = os.path.join(args.cache, 'target.pickle')
                target_vocab, target_data = cached_call(cache_file,
                                                        read_target,
                                                        args.input, args.cache)
            else:
                target_vocab, target_data = read_target(args.input, args.cache)
            et = time.time()
            print("RD target done. {:.3f} [s]".format(et - bt))
            sys.stdout.flush()

            print('Original training data size: %d' % len(source_data))
            train_data = [(s, t)
                          for s, t in six.moves.zip(source_data, target_data)
                          if 0 < len(s) < 50 and 0 < len(t) < 50]
            print('Filtered training data size: %d' % len(train_data))

            #import pdb; pdb.set_trace()

            en_path = os.path.join(args.input, 'dev', 'newstest2013.en')
            source_data = europal.make_dataset(en_path, source_vocab)
            fr_path = os.path.join(args.input, 'dev', 'newstest2013.fr')
            target_data = europal.make_dataset(fr_path, target_vocab)
            assert(len(source_data) == len(target_data))
            test_data = [(s, t) for s, t
                         in six.moves.zip(source_data, target_data)
                         if 0 < len(s) and 0 < len(t)]

            source_ids = {word: index
                          for index, word in enumerate(source_vocab)}
            target_ids = {word: index
                          for index, word in enumerate(target_vocab)}
        else:
            # target_data, source_data = None, None
            train_data, test_data = None, None
            target_ids, source_ids = None, None

    comm.mpi_comm.Barrier()
    # Check file
    for i in range(0, comm.size):
        if comm.rank == i:
            print("Rank {} GPU: {}".format(comm.rank, dev))
        sys.stdout.flush()
        comm.mpi_comm.Barrier()

    # broadcast id- > word dictionary
    source_ids = comm.mpi_comm.bcast(source_ids, root=0)
    target_ids = comm.mpi_comm.bcast(target_ids, root=0)

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    if comm.rank == 0:
        print("target_words : {}".format(len(target_words)))
        print("source_words : {}".format(len(source_words)))

    model = Seq2seq(3, len(source_ids), len(target_ids), args.unit)

    if dev >= 0:
        chainer.cuda.get_device(dev).use()
        model.to_gpu(dev)

    # determine the stop trigger
    m = re.match(r'^(\d+)e$', args.stop)
    if m:
        trigger = chainermn.get_epoch_trigger(int(m.group(1)), train_data, args.batchsize, comm)
    else:
        m = re.match(r'^(\d+)i$', args.stop)
        if m:
            trigger = (int(m.group(1)), 'iteration')
        else:
            if comm.rank == 0:
                sys.stderr.write("Error: unknown stop trigger: {}".format(args.stop))
            exit(-1)

    if comm.rank == 0:
        print("Trigger: {}".format(trigger))

    #optimizer = chainer.optimizers.Adam()
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    # Broadcast dataset
    # Sanity check of train_data

    # Empirically, if the size of train_data goes beyond 14000000,
    # the pickled size exceeds 4GB, which is a limit of data size
    # that MPI can send in a single MPI_Send/MPI_Recv.
    # Thus, we split train_data in a unit of 10000000 and send them
    # separately and join later.

    # Compute the # of scatter_dataset call
    Nmax=1000000
    if comm.rank == 0:
        n_iter = int((len(train_data) + Nmax - 1) / Nmax)
    else:
        n_iter = 0
    n_iter = comm.mpi_comm.bcast(n_iter, root=0)
    recv_train_data = []
    for i in range(0, n_iter):
        beg = i * Nmax
        end = (i+1) * Nmax
        data = train_data[beg:end] if train_data is not None else None
        recv_train_data += chainermn.scatter_dataset(data, comm)
    train_data = recv_train_data
    test_data_all = test_data  # NOQA
    test_data = chainermn.scatter_dataset(test_data, comm)

    train_iter = chainer.iterators.SerialIterator(train_data,
                                                  args.batchsize,
                                                  shuffle=False)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=dev)
    trainer = training.Trainer(updater,
                               trigger,
                               # Use epoch trigger
                               (args.epoch, 'epoch'),
                               out=args.out)

    trainer.extend(chainermn.create_multi_node_evaluator(
        BleuEvaluator(model, test_data, device=dev, comm=comm),
        comm))

    def translate_one(source, target):
        words = europal.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    # @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        translate_one(
            'Who are we ?',
            'Qui sommes-nous?')
        translate_one(
            'And it often costs over a hundred dollars ' +
            'to obtain the required identity card .',
            'Or, il en coûte souvent plus de cent dollars ' +
            'pour obtenir la carte d\'identité requise.')

        source, target = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words.get(i, '') for i in source])
        target = ' '.join([target_words.get(i, '') for i in target])
        translate_one(source, target)

    if comm.rank == 0:
        # trainer.extend(BleuEvaluator(model, test_data_all, dev))
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')),
                       trigger=(1, 'epoch'))

        report = extensions.PrintReport(['epoch',
                                         'iteration',
                                         'main/loss',
                                         # 'validation/main/loss',
                                         'main/perp',
                                         'validation/main/bleu',
                                         # 'validation/main/perp',
                                         'elapsed_time'])
        trainer.extend(report, trigger=(1, 'epoch'))
        # trainer.extend(translate, trigger=(200, 'iteration'))

        # trainer.extend(CalculateBleu(model, test_data),
        #                trigger=(10, 'iteration'))

    comm.mpi_comm.Barrier()
    if comm.rank == 0:
        print('start training')
        sys.stdout.flush()

    trainer.run()


if __name__ == '__main__':
    main()
