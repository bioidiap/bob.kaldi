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
  dubm = bob.kaldi.ubm_train(array, temp_file, num_gauss = 2,
  num_gselect = 2, num_iters = 1)

  assert os.path.exists(dubm)

  
def test_ubm_full_train():

  temp_dubm_file = bob.io.base.test_utils.temporary_filename()
  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

  data = bob.io.audio.reader(sample)
  # MFCC
  array = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
  # Train small diagonal GMM
  dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss = 2,
  num_gselect = 2, num_iters = 1)
  # Train small full GMM
  ubm = bob.kaldi.ubm_full_train(array, temp_dubm_file,
  num_gselect = 2, num_iters = 1)
  
  assert os.path.exists(ubm)

  
def test_ubm_enroll():

  temp_dubm_file = bob.io.base.test_utils.temporary_filename()
  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

  data = bob.io.audio.reader(sample)
  # MFCC
  array = bob.kaldi.mfcc(data.load()[0], data.rate,
  normalization=False)
  # Train small diagonal GMM
  dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss = 2,
  num_gselect = 2, num_iters = 1)
  # Perform MAP adaptation of the GMM
  spk_model = bob.kaldi.ubm_enroll(array, dubm)
  
  assert os.path.exists(spk_model)

def test_gmm_score():

  temp_dubm_file = bob.io.base.test_utils.temporary_filename()
  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

  data = bob.io.audio.reader(sample)
  # MFCC
  array = bob.kaldi.mfcc(data.load()[0], data.rate,
  normalization=False)
  # Train small diagonal GMM
  dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss = 2,
  num_gselect = 2, num_iters = 1)
  # Perform MAP adaptation of the GMM
  spk_model = bob.kaldi.ubm_enroll(array, dubm)
  # GMM scoring
  score = bob.kaldi.gmm_score(array, spk_model, dubm)

  assert np.allclose(score, [ 0.26603000000000065 ])

def test_gmm_score_fast():

  temp_dubm_file = bob.io.base.test_utils.temporary_filename()
  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')

  data = bob.io.audio.reader(sample)
  # MFCC
  array = bob.kaldi.mfcc(data.load()[0], data.rate,
  normalization=False)
  # Train small diagonal GMM
  dubm = bob.kaldi.ubm_train(array, temp_dubm_file, num_gauss = 2,
  num_gselect = 2, num_iters = 1)
  # Perform MAP adaptation of the GMM
  spk_model = bob.kaldi.ubm_enroll(array, dubm)
  # GMM scoring
  score = bob.kaldi.gmm_score_fast(array, spk_model, dubm)

  assert np.allclose(score, [ 0.26603000000000065 ])
