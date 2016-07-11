#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

import os
import tempfile

import scipy.io.wavfile
import numpy as np

from . import io
from . import utils


def mfcc (data, rate):
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

  # creates temporary files that magically disappear after the block ends
  with tempfile.NamedTemporaryFile(suffix='.wav') as wavfile:

    # map into 16-bit range
    maxSample=2**15-1
    data = maxSample*data

    # write down wav file
    scipy.io.wavfile.write(wavfile, rate, np.asarray(data,dtype=np.int16))
    wavfile.flush()

    return mfcc_from_path(wavfile.name)


def mfcc_from_path(filename, channel=0):
  """Computes the MFCCs for a given input signal recorded into a file

  Parameters:

    filename (str): A path to a valid WAV or Sphere file to read data from

    channel (int): The audio channel to read from inside the file


  Returns:

    numpy.ndarray: The MFCCs calculated for the input signal (2D array of
      32-bit floats).


  Raises:

    RuntimeError: if any problem was detected during the conversion.

    IOError: if the binary to be executed does not exist

  """

  name = 'abc'
  binary = utils.kaldi_path(['src', 'featbin', 'compute-mfcc-feats'])
  cmd = [binary]

  # creates temporary files that magically disappear after the block ends
  with tempfile.NamedTemporaryFile(suffix='.scp') as scpfile, \
       tempfile.NamedTemporaryFile(suffix='.ark') as arkfile:

    # indicate to kaldi what to process
    scpfile.write(name + ' ' + filename)
    scpfile.flush()

    # compute features into the ark file
    cmd += [
        'scp:' + scpfile.name,
        'ark:' + arkfile.name,
        ]

    os.system(' '.join(cmd))

    # read temporary ark file into numpy array
    return [mat for name,mat in io.read_mat_ark(arkfile.name)][0]
