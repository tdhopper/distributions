# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
A conjugate model on normally-distributied univariate data in which the
prior on the mean is normally distributed and variance is fixed.

The equations used here are from \cite{murphy2007conjugate}.
Murphy, K. "Conjugate Bayesian analysis of the Gaussian distribution" (2007)
Equation numbers referenced below are from this paper.
"""

import numpy as np

from scipy.stats import norm

from distributions.dbg.random import sample_normal
from distributions.mixins import SharedMixin, GroupIoMixin, SharedIoMixin

NAME = 'NormalNormal'
EXAMPLES = [
    {
        'shared': {'mu': 0., 'sigmasq': 1.},
        'values': [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    },
]
Value = float


class Shared(SharedMixin, SharedIoMixin):
    def __init__(self):
        self.mu = None
        self.sigmasq = None
        self.component_sigmasq = None

    def plus_group(self, group):
        """
        \cite{murphy2007conjugate}, Eqs. 20, 24
        """
        sigmasq_n = 1. / (group.count / self.component_sigmasq + 1. / self.sigmasq)
        mu_n = sigmasq_n * (self.mu / self.sigmasq + group.count * group.mean / self.component_sigmasq)

        post = self.__class__()
        post.mu = mu_n
        post.sigmasq = sigmasq_n
        return post

    def load(self, raw):
        self.mu = float(raw['mu'])
        self.sigmasq = float(raw['sigmasq'])
        self.component_sigmasq = float(raw['component_sigmasq'])

    def dump(self):
        return {
            'mu': self.mu,
            'kappa': self.kappa,
            'sigmasq': self.sigmasq,
            'nu': self.nu,
        }

    def protobuf_load(self, message):
        self.mu = float(message.mu)
        self.kappa = float(message.kappa)
        self.sigmasq = float(message.sigmasq)
        self.nu = float(message.nu)

    def protobuf_dump(self, message):
        message.Clear()
        message.mu = self.mu
        message.kappa = self.kappa
        message.sigmasq = self.sigmasq
        message.nu = self.nu


class Group(GroupIoMixin):
    def __init__(self):
        self.count = None
        self.mean = None
        self.sumsquares = None

    def init(self, shared):
        self.count = 0
        self.mean = 0.
        self.sumsquares = 0.

    def add_value(self, shared, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sumsquares += value**2

    def add_repeated_value(self, shared, value, count):
        self.count += count
        delta = count * value - self.mean
        self.mean += delta / self.count
        self.sumsquares += value**2 * count

    def remove_value(self, shared, value):
        total = self.mean * self.count
        self.count -= 1
        self.sumsquares -= value
        if self.count == 0:
            self.mean = 0.
        else:
            self.mean = (total - value) / self.count

    def merge(self, shared, source):
        count = self.count + source.count
        delta = source.mean - self.mean
        source_part = float(source.count) / count
        self.mean += source_part * delta
        self.count = count
        self.sumsquares += source.sumsquares

    def score_value(self, shared, value):
        """
        \cite{murphy2007conjugate}, Eq. 40
        """
        post = shared.plus_group(self)
        sd = np.sqrt(shared.component_sigmasq + post.sigmasq)
        return norm(post.mu, sd).logpdf(value)

    def score_data(self, shared):
        """
        \cite{murphy2007conjugate}, Eq. 171
        """
        denom = (2 * (self.count * shared.sigmasq + shared.component_sigmasq))
        terms = np.array([np.log(np.sqrt(shared.component_sigmasq)),
                          -np.log((np.sqrt(2*np.pi) *
                                  np.sqrt(shared.component_sigmasq))**self.count *
                                  np.sqrt(self.count*shared.sigmasq + shared.component_sigmasq)),
                          -self.sumsquares * 1. / (2 * shared.component_sigmasq),
                          -shared.mu**2 * 1. / (2 * shared.sigmasq),
                          shared.sigmasq * self.count**2 * self.mean**2 * 1. / shared.component_sigmasq / denom,
                          shared.component_sigmasq * shared.mu**2 * 1. / shared.sigmasq / denom,
                          2. * self.count * self.mean * shared.mu / denom
                          ])
        return np.array(terms).sum()

    def sample_value(self, shared):
        sampler = Sampler()
        sampler.init(shared, self)
        return sampler.eval(shared)

    def load(self, raw):
        self.count = int(raw['count'])
        self.mean = float(raw['mean'])
        self.sumsquares = float(raw['sumsquares'])

    def dump(self):
        return {
            'count': self.count,
            'mean': self.mean,
            'sumsquares': self.sumsquares,
        }

    def protobuf_load(self, message):
        self.count = int(message.count)
        self.mean = float(message.mean)
        self.sumsquares = float(message.sumsquares)

    def protobuf_dump(self, message):
        message.count = self.count
        message.mean = self.mean
        message.sumsquares = self.sumsquares


class Sampler(object):
    def init(self, shared, group=None):
        """
        Draw samples from the marginal posteriors of mu and sigmasq

        \cite{murphy2007conjugate}, Eqs. 17, 20, 24
        """
        post = shared if group is None else shared.plus_group(group)
        self.mu = sample_normal(post.mu, np.sqrt(post.sigmasq))

    def eval(self, shared):
        return sample_normal(self.mu, np.sqrt(shared.component_sigmasq))


def sample_group(shared, size):
    group = Group()
    group.init(shared)
    sampler = Sampler()
    sampler.init(shared, group)
    return [sampler.eval(shared) for _ in xrange(size)]
