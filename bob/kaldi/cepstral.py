#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Milos Cernak <milos.cernak@idiap.ch>
# August 28, 2017

import os

import numpy as np

from . import io
from subprocess import PIPE, Popen
from os.path import isfile
import tempfile
# import shutil
import logging
logger = logging.getLogger(__name__)


def cepstral(data, cepstral_type, rate=8000,
         preemphasis_coefficient=0.97, raw_energy=True, delta_order=2,
         frame_length=25, frame_shift=10, num_ceps=13,
         num_mel_bins=23, cepstral_lifter=22, low_freq=20,
         high_freq=0, dither=1.0, snip_edges=True, normalization=True): 
    """Computes the cepstral (mfcc/plp) features for given speech samples.

    Parameters
    ----------
    data : numpy.ndarray
        A 1D numpy ndarray object containing 64-bit float numbers with
        the audio signal to calculate the cepstral features from. The
        input needs to be normalized between [-1, 1].

    rate : float
        The sampling rate of the input signal in ``data``.

    cepstral_type: str
        The type of cepstral features: mfcc or plp

    preemphasis_coefficient : :obj:`float`, optional
        Coefficient for use in signal preemphasis
    raw_energy : :obj:`bool`, optional
        If true, compute energy before preemphasis and windowing
    delta_order : :obj:`int`, optional
        Add deltas to raw mfcc or plp features
    frame_length : :obj:`int`, optional
        Frame length in milliseconds
    frame_shift : :obj:`int`, optional
        Frame shift in milliseconds
    num_ceps : :obj:`int`, optional
        Number of cepstra in MFCC computation (including C0)
    num_mel_bins : :obj:`int`, optional
        Number of triangular mel-frequency bins
    cepstral_lifter : :obj:`int`, optional
        Constant that controls scaling of MFCCs
    low_freq : :obj:`int`, optional
        Low cutoff frequency for mel bins
    high_freq : :obj:`int`, optional
        High cutoff frequency for mel bins (if < 0, offset from Nyquist)
    dither : :obj:`float`, optional
        Dithering constant (0.0 means no dither)
    snip_edges : :obj:`bool`, optional
        If true, end effects will be handled by outputting only frames
        that completely fit in the file, and the number of frames
        depends on the frame-length.  If false, the number of frames
        depends only on the frame-shift, and we reflect the data at
        the ends. 
    normalization : :obj:`bool`, optional
        If true, the input samples in ``data`` are normalized to [-1, 1].

    Returns
    -------
    numpy.ndarray
        The cepstral features calculated for the input signal (2D
        array of 32-bit floats).

    """

    assert(cepstral_type == 'mfcc' or cepstral_type == 'plp')
    binary1 = 'compute-' + cepstral_type + '-feats'
    cmd1 = [binary1]
    binary2 = 'compute-cmvn-stats'
    cmd2 = [binary2]
    binary3 = 'apply-cmvn'
    cmd3 = [binary3]
    binary4 = 'add-deltas'
    cmd4 = [binary4]

    # compute features plus deltas and sliding cmvn into the ark file
    cmd1 += [
        '--sample-frequency=' + str(rate),
        '--preemphasis-coefficient=' + str(preemphasis_coefficient),
        '--raw-energy=' + str(raw_energy).lower(),
        '--frame-length=' + str(frame_length),
        '--frame-shift=' + str(frame_shift),
        '--num-ceps=' + str(num_ceps),
        '--num-mel-bins=' + str(num_mel_bins),
        '--cepstral-lifter=' + str(cepstral_lifter),
        '--dither=' + str(dither),
        '--snip-edges=' + str(snip_edges).lower(),
        'ark:-',
        'ark:-',
    ]
    cmd4 += [
        '--delta-order=' + str(delta_order),
        'ark:-',
        'ark:-',
    ]

    # import ipdb; ipdb.set_trace()
    if normalization:
        data /= np.max(np.abs(data), axis=0)  # normalize to [-1,1]

    # Compute static features
    with open(os.devnull, "w") as fnull:
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=fnull)

        # write wav file name (as if it were a Kaldi ark file)
        pipe1.stdin.write(b'abc ')
        # write WAV file in 16-bit format
        io.write_wav(pipe1.stdin, data, rate)
        pipe1.stdin.close()

        feats = [mat for name, mat in io.read_mat_ark(pipe1.stdout)][0]

    assert len(feats)

    # Compute and apply CMVN with deltas
    with tempfile.NamedTemporaryFile(suffix='.cmvn') as cmvnfile,\
        open(os.devnull, "w") as fnull:
        
        cmd2 += [
            'ark:-',
            'arkcmvnfile.name',
        ]

        pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=fnull)
        io.write_mat(pipe2.stdin, feats, key=b'abc')
        # pipe2.stdin.close()
        pipe2.communicate()

        cmd3 += [
            'arkcmvnfile.name',
            'ark:-',
            'ark:-',
        ]

        pipe3 = Popen(cmd3, stdin=PIPE, stdout=PIPE, stderr=fnull)
        pipe4 = Popen(cmd4, stdin=pipe3.stdout, stdout=PIPE, stderr=fnull)
        io.write_mat(pipe3.stdin, feats, key=b'abc')
        pipe3.stdin.close()
        
        ret = [mat for name, mat in io.read_mat_ark(pipe4.stdout)][0]

        return ret
