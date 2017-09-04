#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Milos Cernak <milos.cernak@idiap.ch>
# September 1, 2017

'''Tests for Kaldi bindings'''

import pkg_resources
import numpy as np
import bob.io.audio

import bob.kaldi


def test_forward_pass():

    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
    nnetfile   = pkg_resources.resource_filename(__name__, 'dnn/ami.nnet.txt')
    transfile = pkg_resources.resource_filename(__name__,
    'dnn/ami.feature_transform.txt')
    
    reference = pkg_resources.resource_filename(
        __name__, 'data/sample16k-posteriors.txt')

    data = bob.io.audio.reader(sample)

    feats = bob.kaldi.cepstral(data.load()[0], 'mfcc', data.rate,
           normalization=False)

    with open(nnetfile) as nnetf, \
        open(transfile) as trnf:
        dnn = nnetf.read()
        trn = trnf.read()
        ours = bob.kaldi.nnet_forward(feats, dnn, trn)
    
    theirs = np.loadtxt(reference)

    assert ours.shape == theirs.shape

    assert np.allclose(ours, theirs, 1e-03, 1e-05)

def test_compute_dnn_vad():

    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
    reference = pkg_resources.resource_filename(
        __name__, 'data/sample16k-dnn-vad.txt')

    data = bob.io.audio.reader(sample)

    ours = bob.kaldi.compute_dnn_vad(data.load()[0], data.rate)
    theirs = np.loadtxt(reference)

    assert np.allclose(ours, theirs)

    
    
