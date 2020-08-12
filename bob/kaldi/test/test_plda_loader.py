#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Seyyed Saeed Sarfjoo <saeed.sarfjoo@idiap.ch>
# Tue 11 Dec 2018 10:39:15 CET

"""Tests for Kaldi bindings"""

import numpy as np
import pkg_resources

import bob.kaldi
from bob.kaldi.io import read_mat_ark
from bob.kaldi.io import read_plda


def test_plda():
    plda_file = pkg_resources.resource_filename(__name__, "data/plda")
    plda_dic = read_plda(plda_file)
    offset_ = -1.0 * np.dot(plda_dic["transform"], plda_dic["mean"]).T
    assert np.sum(offset_) > 0 and np.sum(offset_) < 1


def test_readmatrix():
    ivector_ark = pkg_resources.resource_filename(__name__, "data/ivector.ark")
    # read vector ark with read_mat function
    d = {key: mat for key, mat in read_mat_ark(ivector_ark)}
    assert len(d) == 281
