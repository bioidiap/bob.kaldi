#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Milos Cernak <milos.cernak@idiap.ch>
# September 1, 2017

'''Tests for Kaldi bindings'''

import pkg_resources
import os.path
import numpy as np
import bob.io.audio

import bob.kaldi


def test_train_mono():

    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
    fstfile   = pkg_resources.resource_filename(__name__, 'hmm/L.fst')
    topofile = pkg_resources.resource_filename(__name__, 'hmm/topo.txt')
    phfile = pkg_resources.resource_filename(__name__, 'hmm/sets.txt')

    # word labels
    uttid='test'
    labels = uttid + ' 27312 27312 27312'
    
    data = bob.io.audio.reader(sample)
    feats = bob.kaldi.cepstral(data.load()[0], 'mfcc', data.rate,
           normalization=False)

    train_set={}
    train_set[uttid]=feats
    with open(topofile) as topof:
        topo = topof.read()
        out = bob.kaldi.train_mono(train_set, labels, fstfile, topo,
                                   phfile , numgauss=2, num_iters=2)

    assert out.find('TransitionModel')
