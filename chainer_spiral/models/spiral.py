from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from builtins import *  # NOQA

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'  # NOQA

import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainerrl.distribution import SoftmaxDistribution
from chainerrl.recurrent import RecurrentChainMixin
from future import standard_library
from chainerrl import v_function
from chainerrl import distribution
from chainerrl import policies

standard_library.install_aliases()  # NOQA


def bw_linear(x_in, x, l):
    return F.matmul(x, l.W)


def bw_convolution(x_in, x, l):
    return F.deconvolution_2d(x, l.W, None, l.stride, l.pad,
                              (x_in.data.shape[2], x_in.data.shape[3]))


def bw_leaky_relu(x_in, x, a):
    return (x_in.data > 0) * x + a * (x_in.data < 0) * x


class SPIRALModel(chainer.Link, RecurrentChainMixin):
    """ SPIRAL Model. """

    def pi_and_v(self, obs):
        """ evaluate the policy and the V-function """
        return NotImplementedError()

    def __call__(self, obs):
        return self.pi_and_v(obs)



class SpiralDiscriminator(chainer.Chain):
    """ Discriminator """

    def __init__(self, imsize, conditional):
        self.imsize = imsize
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_channel = 6 if self.conditional else 3
            self.c1 = L.Convolution2D(in_channel, 16, stride=1, ksize=3, pad=1)
            self.c2 = L.Convolution2D(16, 32, stride=2, ksize=3, pad=1)
            self.c3 = L.Convolution2D(32, 48, stride=2, ksize=2, pad=1)
            self.c4 = L.Convolution2D(48, 48, stride=2, ksize=2, pad=1)
            self.c5 = L.Convolution2D(48, 64, stride=2, ksize=2, pad=1)
            self.c6 = L.Convolution2D(64, 64, stride=2, ksize=2, pad=1)

            self.l7 = L.Linear(3 * 3 * 64, 1)

    def __call__(self, x, conditional_input=None):
        if self.conditional:
            self.x = F.concat((x, conditional_input), axis=1)
        else:
            self.x = x

        self.h1 = F.leaky_relu(self.c1(self.x))
        self.h2 = F.leaky_relu(self.c2(self.h1))
        self.h3 = F.leaky_relu(self.c3(self.h2))
        self.h4 = F.leaky_relu(self.c4(self.h3))
        self.h5 = F.leaky_relu(self.c5(self.h4))
        self.h6 = F.leaky_relu(self.c6(self.h5))
        return self.l7(self.h6.reshape((len(self.h6), 3 * 3 * 64)))

    def differentiable_backward(self, x):
        g = bw_linear(self.h6, x, self.l7)
        g = F.reshape(g, (x.shape[0], 64, 3, 3))
        g = bw_leaky_relu(self.h6, g, 0.2)
        g = bw_convolution(self.h5, g, self.c6)
        g = bw_leaky_relu(self.h5, g, 0.2)
        g = bw_convolution(self.h4, g, self.c5)
        g = bw_leaky_relu(self.h4, g, 0.2)
        g = bw_convolution(self.h3, g, self.c4)
        g = bw_leaky_relu(self.h3, g, 0.2)
        g = bw_convolution(self.h2, g, self.c3)
        g = bw_leaky_relu(self.h2, g, 0.2)
        g = bw_convolution(self.h1, g, self.c2)
        g = bw_leaky_relu(self.h1, g, 0.2)
        g = bw_convolution(self.x, g, self.c1)
        return g


class SpiralModel(chainer.Chain, SPIRALModel, RecurrentChainMixin):
    """ Generator """

    def __init__(self, imsize, action_size, L_stages, conditional):
        self.imsize = imsize
        self.action_size = action_size
        self.f = F.relu  # activation func for encoding part
        self.L_stages = L_stages
        self.conditional = conditional
        super().__init__()
        with self.init_scope():
            in_channel = 6 if self.conditional else 3
            self.c1 = L.Convolution2D(in_channel, 16, stride=1, ksize=3, pad=1)
            self.c2 = L.Convolution2D(16, 32, stride=2, ksize=3, pad=1)
            self.c3 = L.Convolution2D(32, 48, stride=2, ksize=2, pad=1)
            self.c4 = L.Convolution2D(48, 48, stride=2, ksize=2, pad=1)
            self.c5 = L.Convolution2D(48, 64, stride=2, ksize=2, pad=1)
            self.c6 = L.Convolution2D(64, self.L_stages+12, stride=2, ksize=2, pad=1)

            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(48)
            self.bn3 = L.BatchNormalization(48)
            self.bn4 = L.BatchNormalization(64)
            self.bn5 = L.BatchNormalization(self.L_stages+12)

            self.v = v_function.FCVFunction(3 * 3 * (self.L_stages+12))

            self.dc1 = L.Convolution1D(1, 16, stride=1, ksize=3)
            self.dc2 = L.Convolution1D(16, 32, stride=1, ksize=3)
            self.dc3 = L.Convolution1D(32, 48, stride=1, ksize=3)
            self.dc4 = L.Convolution1D(48, 48, stride=1, ksize=3)
            self.dc5 = L.Convolution1D(48, 64, stride=1, ksize=3)
            self.dc6 = L.Convolution1D(64, action_size, stride=1, ksize=3)

            self.dbn1 = L.BatchNormalization(16)
            self.dbn2 = L.BatchNormalization(32)
            self.dbn3 = L.BatchNormalization(48)
            self.dbn4 = L.BatchNormalization(48)
            self.dbn5 = L.BatchNormalization(64)


    def pi_and_v(self, state, conditional_input=None):
        if self.conditional:
            self.x = F.concat((state, conditional_input), axis=1)
        else:
            self.x = state

        self.h1 = self.f(self.c1(self.x))
        self.h2 = self.f(self.bn1(self.c2(self.h1)))
        self.h3 = self.f(self.bn2(self.c3(self.h2)))
        self.h4 = self.f(self.bn3(self.c4(self.h3)))
        self.h5 = self.f(self.bn4(self.c5(self.h4)))
        self.h6 = self.f(self.bn5(self.c6(self.h5)))

        return self.compute_policy(self.h6), self.compute_value(self.h6)

    def compute_policy(self, h):
        h = F.average_pooling_2d(h, 3)
        h = h.reshape((len(h), 1, self.L_stages+12))
        h = self.f(self.dbn1(self.dc1(h)))
        h = self.f(self.dbn2(self.dc2(h)))
        h = self.f(self.dbn3(self.dc3(h)))
        h = self.f(self.dbn4(self.dc4(h)))
        h = self.f(self.dbn5(self.dc5(h)))
        h = self.dc6(h)

        probs = []
        acts = []

        for i in range(self.action_size):
            p = SoftmaxDistribution(h[:, i, :])
            a = p.sample()
            probs.append(p)
            acts.append(a)

        return probs, acts

    def compute_value(self, h):
        return self.v(h)
