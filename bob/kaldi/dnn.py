#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Milos Cernak <milos.cernak@idiap.ch>
# August 31, 2017

import os

import numpy as np

from . import io
from subprocess import PIPE, Popen
from os.path import isfile
import tempfile
# import shutil
import logging
import pkg_resources

import bob.kaldi

logger = logging.getLogger(__name__)


def nnet_forward(feats, nnet, feats_transform='', apply_log=False,
                 no_softmax=False, prior_floor=1e-10, prior_scale=1,
                 use_gpu=False):
    """Computes the forward pass for given features.

    Parameters
    ----------
    feats: numpy.ndarray
        The input cepstral features (2D array of 32-bit floats).
    nnet: str
        The neural network

    feats_transform : :obj:`str`, optional
        The input feature transform for ``feats``.
    apply_log : :obj:`bool`, optional
        Transform NN output by log().
    no_softmax : :obj:`bool`, optional
        Removes the last component with Softmax.
    prior_floor : :obj:`float`, optional
        Flooring constant for prior probability.
    prior_scale : :obj:`float`, optional
        Scaling factor to be applied on pdf-log-priors.
    use_gpu : :obj:`bool`, optional
        Compute forward pass on GPU.

    Returns
    -------
    numpy.ndarray
        The posterior features.

    """

    binary1 = 'nnet-forward'
    cmd1 = [binary1]

    cmd1 += [
        '--apply-log=' + str(apply_log).lower(),
        '--no-softmax=' + str(no_softmax).lower(),
        '--prior-floor=' + str(prior_floor),
        '--prior-scale=' + str(prior_scale),
        '--use-gpu=' + str(use_gpu).lower(),
    ]
        
    # save nnet model to a file
    with tempfile.NamedTemporaryFile(
            delete=False, suffix='.nnet') as dnn:
        with open(dnn.name, 'wt') as fp:
            fp.write(nnet)

    if feats_transform != '':
        # save nnet transform model to a file
        with tempfile.NamedTemporaryFile(
                delete=False, suffix='.nnet') as transf:
            with open(transf.name, 'wt') as fp:
                fp.write(feats_transform)
                
        cmd1 += [
            '--feature-transform=' + transf.name,
        ]

    cmd1 += [
        dnn.name,
        'ark:-',
        'ark:-',
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        io.write_mat(pipe1.stdin, feats, key=b'abc')
        pipe1.stdin.close()
        # pipe1.communicate()

        posts = [mat for name, mat in io.read_mat_ark(pipe1.stdout)][0]

        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

    os.unlink(dnn.name)
    if feats_transform != '':
        os.unlink(transf.name)
    
    return posts

def compute_dnn_vad(samples, rate, silence_threshold=0.9, posterior=0):
    """Performs Voice Activity Detection on a Kaldi feature matrix

    Parameters
    ----------
    feats : numpy.ndarray
        A 2-D numpy array, with log-energy being in the first
        component of each feature vector
    rate : float
        The sampling rate of the input signal in ``samples``.
    silence_threshold: :obj:`float`, optional
        Silence threshold to be used for silence posterior
        evaluation. 
    posterior: :obj:`int`, optional
        Index of posterior feature to be used for detection. Useful
        ones are 0, 1 and 2, for silence, laughter and
        noise,respectively.

    Returns
    -------
    numpy.ndarray
        The labels [1/0] of voiced features (1D array of floats).
    """

    nnetfile   = pkg_resources.resource_filename(__name__,
    'test/dnn/ami.nnet.txt')
    transfile = pkg_resources.resource_filename(__name__,
    'test/dnn/ami.feature_transform.txt')

    feats = bob.kaldi.cepstral(samples, 'mfcc', rate,
    normalization=False)

    with open(nnetfile) as nnetf, \
        open(transfile) as trnf:
        dnn = nnetf.read()
        trn = trnf.read()
        post = bob.kaldi.nnet_forward(feats, dnn, trn)

    vad = []
    for row in post:
        if row[posterior] > silence_threshold:
            vad.append(0.0)
        else:
            vad.append(1.0)

    return vad
