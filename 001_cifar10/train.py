#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np

from VGG import VGG

def init_args():

    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    return args


def load_data():
    import cPickle

    with open('/home/gachiemchiep/workspace/DL/data/cifar-100-python/train', 'rb') as fo:
        train_dict = cPickle.load(fo)

    with open('/home/gachiemchiep/workspace/DL/data/cifar-100-python/test', 'rb') as fo:
        test_dict = cPickle.load(fo)

    train_imgs = np.array(train_dict['data'].reshape(-1, 3, 32, 32), dtype=np.float32)
    train_labels = np.array(train_dict['fine_labels'], dtype=np.int32)

    test_imgs = np.array(train_dict['data'].reshape(-1, 3, 32, 32), dtype=np.float32)
    test_labels = np.array(train_dict['fine_labels'], dtype=np.int32)

    train = chainer.datasets.tuple_dataset.TupleDataset(train_imgs, train_labels)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_imgs, test_labels)

    labels = len(np.unique(test_labels))

    return train, test, labels


def main():

    args = init_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # 1. Load data
    # from https://www.cs.toronto.edu/~kriz/cifar.html
    train, test, class_labels = load_data()

    # 2. Load network definition
    model = L.Classifier(VGG(class_labels))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # 3. Network optimizer
    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # 4. Iterator (train, test)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # 5. Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Save snapshot every 10 epoch
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),trigger=(25, 'epoch'))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()