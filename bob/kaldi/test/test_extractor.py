#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

"""Tests for Kaldi bindings"""

import numpy as np
import pkg_resources

import bob.io.audio
import bob.kaldi


def test_mfcc():

    sample = pkg_resources.resource_filename(__name__, "data/sample16k.wav")
    reference = pkg_resources.resource_filename(__name__, "data/sample16k-mfcc.txt")

    data = bob.io.audio.reader(sample)

    ours = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
    theirs = np.loadtxt(reference)

    assert ours.shape == theirs.shape

    np.testing.assert_allclose(ours, theirs, 1e-02, 1e-02)


def test_mfcc_from_path():

    sample = pkg_resources.resource_filename(__name__, "data/sample16k.wav")
    reference = pkg_resources.resource_filename(__name__, "data/sample16k-mfcc.txt")

    ours = bob.kaldi.mfcc_from_path(sample)
    theirs = np.loadtxt(reference)

    assert ours.shape == theirs.shape

    np.testing.assert_allclose(ours, theirs, 1e-02, 1e-02)


def test_compute_vad():

    sample = pkg_resources.resource_filename(__name__, "data/sample16k.wav")
    reference = pkg_resources.resource_filename(__name__, "data/sample16k-vad.txt")

    data = bob.io.audio.reader(sample)

    ours = bob.kaldi.compute_vad(data.load()[0], data.rate)
    theirs = np.loadtxt(reference)

    np.testing.assert_allclose(ours, theirs)


def test_cepstral_mfcc():

    sample = pkg_resources.resource_filename(__name__, "data/sample16k.wav")
    reference = pkg_resources.resource_filename(
        __name__, "data/sample16k-cepstral-mfcc.txt"
    )

    data = bob.io.audio.reader(sample)

    ours = bob.kaldi.cepstral(data.load()[0], "mfcc", data.rate, normalization=False)
    theirs = np.loadtxt(reference)

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, 1e-02, 1e-02)


def test_cepstral_plp():

    sample = pkg_resources.resource_filename(__name__, "data/sample16k.wav")
    reference = pkg_resources.resource_filename(
        __name__, "data/sample16k-cepstral-plp.txt"
    )

    data = bob.io.audio.reader(sample)

    ours = bob.kaldi.cepstral(data.load()[0], "plp", data.rate, normalization=False)
    theirs = np.loadtxt(reference)

    assert ours.shape == theirs.shape

    np.testing.assert_allclose(ours, theirs, 1e-02, 1e-02)
