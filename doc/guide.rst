.. py:currentmodule:: bob.kaldi

.. testsetup:: *

   from __future__ import print_function
   import pkg_resources
   import bob.kaldi
   import bob.io.audio
   import tempfile
   import os

=======================
 Using Kaldi in Python
=======================

MFCC Extraction
---------------

Two functions are implemented to extract MFCC features
:py:func:`bob.kaldi.mfcc` and :py:func:`bob.kaldi.mfcc_from_path`. The former
function accepts the speech samples as :obj:`numpy.ndarray`, whereas the latter
the filename as :obj:`str`:

1. :py:func:`bob.kaldi.mfcc`

   .. doctest::

      >>> sample = pkg_resources.resource_filename('bob.kaldi', 'test/data/sample16k.wav')
      >>> data = bob.io.audio.reader(sample)
      >>> feat = bob.kaldi.mfcc(data.load()[0], data.rate, normalization=False)
      >>> print (feat.shape)
      (317, 39)

2. :py:func:`bob.kaldi.mfcc_from_path`

   .. doctest::

      >>> feat = bob.kaldi.mfcc_from_path(sample)
      >>> print (feat.shape)
      (317, 39)

Voice Activity Detection (VAD)
------------------------------

A simple energy-based VAD is implemented in :py:func:`bob.kaldi.compute_vad`.
The function expects the speech samples as :obj:`numpy.ndarray` and the sampling
rate as :obj:`float`, and returns an array of VAD labels :obj:`numpy.ndarray`
with the labels of 0 (zero) or 1 (one) per speech frame:

.. doctest::

   >>> VAD_labels = bob.kaldi.compute_vad(data.load()[0], data.rate)
   >>> print (len(VAD_labels))
   317


UBM training and evaluation
---------------------------

Both diagonal and full covariance Universal Background Models (UBMs)
are supported, speakers can be enrolled and scored:

.. doctest::

  >>> # Train small diagonall GMM
  >>> diag_gmm_file = tempfile.NamedTemporaryFile()
  >>> full_gmm_file = tempfile.NamedTemporaryFile()
  >>> dubm = bob.kaldi.ubm_train(feat, diag_gmm_file.name, num_gauss=2, num_gselect=2, num_iters=2)
  >>> # Train small full GMM
  >>> ubm = bob.kaldi.ubm_full_train(feat, dubm, full_gmm_file.name, num_gselect=2, num_iters=2)
  >>> # Enrollement - MAP adaptation of the UBM-GMM
  >>> spk_model = bob.kaldi.ubm_enroll(feat, dubm)
  >>> # GMM scoring
  >>> score = bob.kaldi.gmm_score(feat, spk_model, dubm)
  >>> print ('%.3f' % score)
  0.282

Following guide describes how to run whole speaker recognition experiments:

1. To run the UBM-GMM with MAP adaptation speaker recognition experiment, run:

.. code-block:: sh

	verify.py -d 'mobio-audio-male' -p 'energy-2gauss' -e 'mfcc-kaldi' -a 'gmm-kaldi' -s exp-gmm-kaldi --groups {dev,eval} -R '/your/work/directory/' -T '/your/temp/directory' -vv

2. To run the ivector+plda speaker recognition experiment, run:

.. code-block:: sh

	verify.py -d 'mobio-audio-male' -p 'energy-2gauss' -e 'mfcc-kaldi' -a 'ivector-plda-kaldi' -s exp-ivector-plda-kaldi --groups {dev,eval} -R '/your/work/directory/' -T '/your/temp/directory' -vv

3. Results:

+---------------------------------------------------+--------+--------+
| Experiment description                            |    EER |   HTER |
+---------------------------------------------------+--------+--------+
| -e 'mfcc-kaldi', -a 'gmm-kadi', 100GMM            | 18.53% | 14.52% |
+---------------------------------------------------+--------+--------+
| -e 'mfcc-kaldi', -a 'gmm-kadi', 512GMM            | 17.51% | 12.44% |
+---------------------------------------------------+--------+--------+
| -e 'mfcc-kaldi', -a 'ivector-plda-kaldi', 64GMM   | 12.26% | 11.97% |
+---------------------------------------------------+--------+--------+
| -e 'mfcc-kaldi', -a 'ivector-plda-kaldi', 256GMM  | 11.35% | 11.46% |
+---------------------------------------------------+--------+--------+


.. include:: links.rst
