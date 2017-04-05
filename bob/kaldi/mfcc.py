#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

import os

import numpy as np

from . import io

from subprocess import PIPE, Popen
# import subprocess
from os.path import join
import tempfile

import logging
logger = logging.getLogger("bob.kaldi")

# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE,SIG_DFL)

def mfcc (data, rate=8000, preemphasis_coefficient=0.97, raw_energy=True, frame_length=25, frame_shift=10, num_ceps=13, num_mel_bins=23, cepstral_lifter=22, low_freq=20, high_freq=0, dither=1.0, snip_edges=True, normalization=True):
  """Computes the MFCCs for a given input signal

  Parameters:

    data (numpy.ndarray): A 1D numpy ndarray object containing 64-bit float
      numbers with the audio signal to calculate the MFCCs from. The input
      needs to be normalized between [-1, 1].

    rate (float): The sampling rate of the input signal in ``data``.


  Returns:

    numpy.ndarray: The MFCCs calculated for the input signal (2D array of
      32-bit floats).


  Raises:

    RuntimeError: if any problem was detected during the conversion.

    IOError: if the binary to be executed does not exist

  """

  name = 'abc'
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
    data /= np.max(np.abs(data),axis=0) # normalize to [-1,1]

  with open(os.devnull, "w") as fnull: 
    pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=fnull)
    pipe2 = Popen (cmd2, stdout=PIPE, stdin=pipe1.stdout, stderr=fnull)
    pipe3 = Popen (cmd3, stdout=PIPE, stdin=pipe2.stdout, stderr=fnull)

    # write wav file name (as if it were a Kaldi ark file)
    pipe1.stdin.write (name + ' ')
    # write WAV file in 16-bit format
    io.write_wav (pipe1.stdin, data, rate)
    pipe1.stdin.close()

    # # wait for piped execution to finish
    # pipe3.communicate()

    # read ark from pipe3.stdout
    ret = [mat for name,mat in io.read_mat_ark(pipe3.stdout)][0]
    return ret


def mfcc_from_path(filename, channel=0, preemphasis_coefficient=0.97, raw_energy=True, frame_length=25, frame_shift=10, num_ceps=13, num_mel_bins=23, cepstral_lifter=22, low_freq=20, high_freq=0, dither=1.0, snip_edges=True):
  """Computes the MFCCs for a given input signal recorded into a file

  Parameters:

    filename (str): A path to a valid WAV or NIST Sphere file to read data from

    channel (int): The audio channel to read from inside the file


  Returns:

    numpy.ndarray: The MFCCs calculated for the input signal (2D array of
      32-bit floats).


  Raises:

    RuntimeError: if any problem was detected during the conversion.

    IOError: if the binary to be executed does not exist

  """

  name = 'abc'
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
  with open(os.devnull, "w") as fnull:
    pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=fnull)
    pipe2 = Popen (cmd2, stdout=PIPE, stdin=pipe1.stdout, stderr=fnull)
    pipe3 = Popen (cmd3, stdout=PIPE, stdin=pipe2.stdout, stderr=fnull)

    # write scp file into pipe.stdin
    pipe1.stdin.write (name + ' ' + filename)
    pipe1.stdin.close()
    # pipe3.communicate()

    # read ark from pipe3.stdout
    ret = [mat for name,mat in io.read_mat_ark(pipe3.stdout)][0]
    return ret

# def compute_vad(feats, vad_energy_mean_scale=0.5, vad_energy_threshold=5, vad_frames_context=0, vad_proportion_threshold=0.6):
#   """Computes speech/non-speech segments given a Kaldi feature matrix

#   Parameters:

#     feats (matrix): A 2-D numpy array, with log-energy being in the first component of each feature vector


#   Returns:

#     A list of speech segments as a int32 numpy array with start and end times

#   Raises:

#     RuntimeError: if any problem was detected during the conversion.

#     IOError: if the binary to be executed does not exist

#   """

#   name = 'abc'
#   binary1 = utils.kaldi_path(['src', 'ivectorbin', 'compute-vad'])
#   cmd1 = [binary1]

#   # compute features into the ark file
#   cmd1 += [
#       '--vad-energy-mean-scale=' + str(vad_energy_mean_scale),
#       '--vad-energy-threshold=' + str(vad_energy_threshold),
#       '--vad-frames-context=' + str(vad_frames_context),
#       '--vad-proportion-threshold=' + str(vad_proportion_threshold),
#       'ark:-',
#       'ark:-',
#       ]

#   with tempfile.NamedTemporaryFile(suffix='.seg') as segfile: 
#     binary2 = utils.kaldi_path(['src', 'ivectorbin', 'create-split-from-vad'])
#     cmd2 = [binary2]

#     cmd2 += [
#       'ark:-',
#       segfile.name,
#     ]

#     with open(os.devnull, "w") as fnull:
#       # pipe1 numpy matrix -> compute-vad
#       pipe1 = Popen(cmd1, stdout=PIPE, stdin=PIPE, stderr=fnull)
#       pipe2 = Popen(cmd2, stdout=PIPE, stdin=pipe1.stdout, stderr=fnull)

#       # write ark file into pipe.stdin
#       io.write_mat(pipe1.stdin, feats, key='abc')
#       pipe1.stdin.close()

#       # wait for piped execution to finish
#       pipe2.communicate()

#       # segfile should have the segmented output. read the file
#       segs = []
#       with open(segfile.name) as fp:
#         for l in fp.readlines():
#           start, end = l.split()[2:]
#           segs.append([start, end])
#       return np.array(segs, dtype='int32')
