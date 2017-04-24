.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Mon 11 Jul 2016 10:32:01 CEST
..
.. Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland

.. _bob.kaldi:


======================
 Bob/Kaldi Extensions
======================

.. todolist::

This module contains information on how to build and maintain |project|
Kaldi_ extensions written in pure Python or a mix of C/C++ and Python.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   py_api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


MFCC Extraction
---------------

Two functions are implemented to extract MFCC features
`bob.kaldi.mfcc` and `bob.kaldi.mfcc_from_path`. The former function
accepts the speech samples as `numpy.ndarray`, whereas the latter the
filename as `str`. Both functions return the features as
`numpy.ndarray`:

1. `bob.kaldi.mfcc`
   .. doctest::

      """
      This is the example mfcc module.

      >>> array = mfcc('data/sample16k.wav')
      """
      
      def mfcc(sample):
          import pkg_resources
	  import bob.io.audio
	  
	  data = bob.io.audio.reader(sample)
	  
	  mfcc = bob.kaldi.mfcc(data.load()[0], data.rate,
	  normalization=False)

      if __name__ == "__main__":
          import doctest
	  doctest.testmod()
   
   
2. `bob.kaldi.mfcc_from_path`

   .. doctest::
      
      """
      This is the example mfcc module.

      >>> array = mfcc_from_path('data/sample16k.wav')
      """

      def mfcc_from_path(sample):
          import pkg_resources
	  import bob.io.audio

	  mfcc = bob.kaldi.mfcc_from_path(sample)

      if __name__ == "__main__":
          import doctest
	  doctest.testmod()
	  
====================
 Speaker recognition
====================
	   
		   
UBM training and evaluation
---------------------------

Both diagonal and full covariance Universal Background Models (UBMs)
are supported. Speakers can be enrolled and evaluated.

Following guide describes how to run whole speaker recognition experiments:

1. To run the UBM-GMM with MAP adaptation speaker recognition experiment, run:

.. code-block:: sh
		
	./bin/verify.py -d 'mobio-audio-male' -p 'energy-2gauss' -e 'mfcc-kaldi' -a 'gmm-kaldi' -s exp-gmm-kaldi --groups {dev,eval} -R '/your/work/directory/' -T '/your/temp/directory' -vv

2. To run the ivector+plda speaker recognition experiment, run:

.. code-block:: sh
		
	./bin/verify.py -d 'mobio-audio-male' -p 'energy-2gauss' -e 'mfcc-kaldi' -a 'ivector-plda-kaldi' -s exp-ivector-plda-kaldi --groups {dev,eval} -R '/your/work/directory/' -T '/your/temp/directory' -vv

3. Results:

+---------------------------------------------------+--------+--------+
| Experiment description                            |    EER |   HTER |
+---------------------------------------------------+--------+--------+
| -e 'mfcc-kaldi', -a 'gmm-kadi', 100GMM            | 18.53% | 14.52% |
| -e 'mfcc-kaldi', -a 'gmm-kadi', 512GMM            | 17.51% | 12.44% |
| -e 'mfcc-kaldi', -a 'ivector-plda-kaldi', 64GMM   | 12.26% | 11.97% |
| -e 'mfcc-kaldi', -a 'ivector-plda-kaldi', 256GMM  | 11.35% | 11.46% |
+---------------------------------------------------+--------+--------+


.. include:: links.rst
