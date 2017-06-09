#!/usr/bin/env python
#
# Milos Cernak <milos.cernak@idiap.ch>
# March 1, 2017
#

'''Tests for Kaldi bindings'''

import pkg_resources
import numpy as np
import bob.io.audio
import bob.io.base.test_utils
import os

import bob.kaldi


def test_ubm_train():

    temp_file = bob.io.base.test_utils.temporary_filename()
    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate,
                           normalization=False)
    # Train small diagonall GMM
    dubm = bob.kaldi.ubm_train(array, temp_file, num_gauss=2,
                               num_gselect=2, num_iters=2)

    # assert os.path.exists(dubm)
    assert dubm.find('DiagGMM')


def test_ubm_full_train():

    temp_dubm_file = bob.io.base.test_utils.temporary_filename()
    temp_fubm_file = bob.io.base.test_utils.temporary_filename()
    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
    # Train small diagonal GMM
    dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss=2,
                               num_gselect=2, num_iters=2)
    # Train small full GMM
    fubm = bob.kaldi.ubm_full_train(array, dubm, temp_fubm_file,
                                   num_gselect=2, num_iters=2)

    assert fubm.find('FullGMM')


def test_ubm_enroll():

    temp_dubm_file = bob.io.base.test_utils.temporary_filename()
    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate,
                           normalization=False)
    # Train small diagonal GMM
    dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss=2,
                               num_gselect=2, num_iters=2)
    # Perform MAP adaptation of the GMM
    spk_model = bob.kaldi.ubm_enroll(array, dubm)

    # assert os.path.exists(spk_model)
    assert spk_model.find('DiagGMM')

def test_gmm_score():

    temp_dubm_file = bob.io.base.test_utils.temporary_filename()
    sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

    data = bob.io.audio.reader(sample)
    # MFCC
    array = bob.kaldi.mfcc(data.load()[0], data.rate,
                           normalization=False)
    # Train small diagonal GMM
    dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss=2,
                               num_gselect=2, num_iters=2)
    # Perform MAP adaptation of the GMM
    spk_model = bob.kaldi.ubm_enroll(array, dubm)
    # GMM scoring
    score = bob.kaldi.gmm_score(array, spk_model, dubm)

    assert np.allclose(score, [0.28216], 1e-03, 1e-05)

# def test_gmm_score_fast():

#   temp_dubm_file = bob.io.base.test_utils.temporary_filename()
#   sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

#   data = bob.io.audio.reader(sample)
#   # MFCC
#   array = bob.kaldi.mfcc(data.load()[0], data.rate,
#   normalization=False)
#   # Train small diagonal GMM
#   dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss = 2,
#                              num_gselect = 2, num_iters = 2)
#   # Perform MAP adaptation of the GMM
#   spk_model = bob.kaldi.ubm_enroll(array, dubm)
#   # GMM scoring
#   score = bob.kaldi.gmm_score_fast(array, spk_model, dubm)

#   assert np.allclose(score, [ 0.282168 ])
