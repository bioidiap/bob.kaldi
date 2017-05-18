#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

import os

import numpy as np

from . import io
from subprocess import PIPE, Popen
from os.path import isfile
import tempfile
import logging
logger = logging.getLogger(__name__)


def mfcc(data, rate=8000, preemphasis_coefficient=0.97, raw_energy=True,
         frame_length=25, frame_shift=10, num_ceps=13, num_mel_bins=23,
         cepstral_lifter=22, low_freq=20, high_freq=0, dither=1.0,
         snip_edges=True, normalization=True):
    """Computes the MFCCs for given speech samples.

    Parameters
    ----------
    data : numpy.ndarray
        A 1D numpy ndarray object containing 64-bit float
        numbers with the audio signal to calculate the MFCCs from. The input
        needs to be normalized between [-1, 1].

    rate : float
        The sampling rate of the input signal in ``data``.

    preemphasis_coefficient : :obj:`float`, optional
        Coefficient for use in signal preemphasis
    raw_energy : :obj:`bool`, optional
        If true, compute energy before preemphasis and windowing
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
        The MFCCs calculated for the input signal (2D array of
        32-bit floats).

    """
    
    binary1 = 'compute-mfcc-feats'
    cmd1 = [binary1]
    binary2 = 'add-deltas'
    cmd2 = [binary2]
    binary3 = 'apply-cmvn-sliding'
    cmd3 = [binary3]

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
    cmd2 += [
        'ark:-',
        'ark:-',
    ]
    cmd3 += [
        '--norm-vars=false',
        '--center=true',
        '--cmn-window=300',
        'ark:-',
        'ark:-',
    ]

    # import ipdb; ipdb.set_trace()
    if normalization:
        data /= np.max(np.abs(data), axis=0)  # normalize to [-1,1]

    with open(os.devnull, "w") as fnull:
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=fnull)
        pipe2 = Popen(cmd2, stdout=PIPE, stdin=pipe1.stdout, stderr=fnull)
        pipe3 = Popen(cmd3, stdout=PIPE, stdin=pipe2.stdout, stderr=fnull)

        # write wav file name (as if it were a Kaldi ark file)
        pipe1.stdin.write(b'abc ')
        # write WAV file in 16-bit format
        io.write_wav(pipe1.stdin, data, rate)
        pipe1.stdin.close()

        ret = [mat for name, mat in io.read_mat_ark(pipe3.stdout)][0]
        return ret

def mfcc_from_path(filename, channel=0, preemphasis_coefficient=0.97,
                   raw_energy=True, frame_length=25, frame_shift=10,
                   num_ceps=13, num_mel_bins=23, cepstral_lifter=22,
                   low_freq=20, high_freq=0, dither=1.0, snip_edges=True):
    """Computes the MFCCs for a given input signal recorded into a file

    Parameters
    ----------
    filename : str
        A path to a valid WAV or NIST Sphere file to read data from

    channel : int
        The audio channel to read from inside the file

    preemphasis_coefficient : :obj:`float`, optional
        Coefficient for use in signal preemphasis
    raw_energy : :obj:`bool`, optional
        If true, compute energy before preemphasis and windowing
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
        the ends

    Returns
    -------
    numpy.ndarray
        The MFCCs calculated for the input signal (2D array of
        32-bit floats).

    """

    binary1 = 'compute-mfcc-feats'
    cmd1 = [binary1]
    binary2 = 'add-deltas'
    cmd2 = [binary2]
    binary3 = 'apply-cmvn-sliding'
    cmd3 = [binary3]

    # compute features into the ark file
    cmd1 += [
        '--channel=' + str(channel),
        '--preemphasis-coefficient=' + str(preemphasis_coefficient),
        '--raw-energy=' + str(raw_energy).lower(),
        '--frame-length=' + str(frame_length),
        '--frame-shift=' + str(frame_shift),
        '--num-ceps=' + str(num_ceps),
        '--num-mel-bins=' + str(num_mel_bins),
        '--cepstral-lifter=' + str(cepstral_lifter),
        '--dither=' + str(dither),
        '--snip-edges=' + str(snip_edges).lower(),
        'scp:-',
        'ark:-',
    ]
    cmd2 += [
        'ark:-',
        'ark:-',
    ]
    cmd3 += [
        '--norm-vars=false',
        '--center=true',
        '--cmn-window=300',
        'ark:-',
        'ark:-',
    ]

    # import ipdb; ipdb.set_trace()
    assert isfile(filename)

    with open(os.devnull, "w") as fnull:
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=fnull)
        pipe2 = Popen(cmd2, stdout=PIPE, stdin=pipe1.stdout, stderr=fnull)
        pipe3 = Popen(cmd3, stdout=PIPE, stdin=pipe2.stdout, stderr=fnull)

        # write scp file into pipe.stdin
        strwrite = 'abc ' + filename
        pipe1.stdin.write(strwrite.encode('utf-8'))
        pipe1.stdin.close()
        # pipe3.communicate()

        # read ark from pipe3.stdout
        ret = [mat for name, mat in io.read_mat_ark(pipe3.stdout)][0]
        return ret

def compute_vad(samples, rate, vad_energy_mean_scale=0.5, vad_energy_th=5,
    vad_frames_context=0, vad_proportion_th=0.6):
    """Performs Voice Activity Detection on a Kaldi feature matrix

    Parameters
    ----------
    feats : numpy.ndarray
        A 2-D numpy array, with log-energy being in the first
        component of each feature vector
    rate : float
        The sampling rate of the input signal in ``samples``.
    vad_energy_mean_scale: :obj:`float`, optional
        If this is set to s, to get the actual threshold we let m be the mean
        log-energy of the file, and use s*m + vad-energy-th
    vad_energy_th: :obj:`float`, optional
        Constant term in energy threshold for MFCC0 for VAD.
    vad_frames_context: :obj:`int`, optional
        Number of frames of context on each side of central frame,
        in window for which energy is monitored 
    vad_proportion_th: :obj:`float`, optional
        Parameter controlling the proportion of frames within the window that
        need to have more energy than the threshold

    Returns
    -------

    numpy.ndarray
        The labels [1/0] of voiced features (1D array of floats).
    """

    binary1 = 'compute-mfcc-feats'
    cmd1 = [binary1]
    binary2 = 'compute-vad'
    cmd2 = [binary2]

    cmd1 += [
        '--sample-frequency=' + str(rate),
        'ark:-',
        'ark:-',
    ]
    cmd2 += [
        '--vad-energy-mean-scale=' + str(vad_energy_mean_scale),
        '--vad-energy-threshold=' + str(vad_energy_th),
        '--vad-frames-context=' + str(vad_frames_context),
        '--vad-proportion-threshold=' + str(vad_proportion_th),
        'ark:-',
        'ark:-',
    ]

    samples /= np.max(np.abs(samples), axis=0)  # normalize to [-1,1]

    with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe2 = Popen(cmd2, stdin=pipe1.stdout, stdout=PIPE, stderr=logfile)

        pipe1.stdin.write(b'abc ')
        io.write_wav(pipe1.stdin, samples, rate)
        pipe1.stdin.close()

        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

        # read ark from pipe2.stdout
        ret = [mat for name, mat in io.read_vec_flt_ark(pipe2.stdout)][0]
        return ret
