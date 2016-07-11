#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

'''Tests for Kaldi bindings'''

import pkg_resources
import numpy


from .mfcc import mfcc, mfcc_from_path


def test_mfcc_from_path():

  sample = pkg_resources.resource_filename(__name__, 'data/sample16k.wav')
  reference = pkg_resources.resource_filename(__name__, 'data/sample16k-mfcc.txt')

  ours = mfcc_from_path(sample)
  theirs = numpy.fromfile(reference)

  import ipdb; ipdb.set_trace()
