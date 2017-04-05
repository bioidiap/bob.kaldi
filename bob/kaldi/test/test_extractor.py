#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

'''Tests for Kaldi bindings'''

import pkg_resources
import numpy as np
import bob.io.audio
import io

import bob.kaldi

def test_mfcc():

  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
  reference = pkg_resources.resource_filename(__name__, 'data/sample16k-mfcc.txt')


  data = bob.io.audio.reader(sample)

  ours = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
  theirs = np.loadtxt(reference)

  assert ours.shape == theirs.shape

  assert np.allclose(ours, theirs, 1e-03, 1e-05)

def test_mfcc_from_path():

  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
  reference = pkg_resources.resource_filename(__name__, 'data/sample16k-mfcc.txt')

  ours = bob.kaldi.mfcc_from_path(sample)
  theirs = np.loadtxt(reference)

  assert ours.shape == theirs.shape

  assert np.allclose(ours, theirs, 1e-03, 1e-05)


# def test_compute_vad():

#   refseg = pkg_resources.resource_filename(__name__, 'data/sample16k.seg')

#   # read and parse reference segmentation file into numpy array of int32
#   segsref = []
#   with open(refseg) as fp:
#     for l in fp.readlines():
#       l = l.strip()
#       s = l.split()
#       start = int(s[2])
#       end = int(s[3])
#       segsref.append([ start, end ])
#   segsref = np.array(segsref, dtype='int32')

#   feats = [mat for name,mat in io.read_mat_ark( pkg_resources.resource_filename(__name__,'data/sample16k.ark') )][0]

#   segs = bob.kaldi.compute_vad(feats)

#   assert np.array_equal(segs,segsref)
