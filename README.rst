.. vim: set fileencoding=utf-8 :
.. Milos Cernak <milos.cernak@idiap.ch>
.. Tue Apr  4 15:28:26 CEST 2017

.. image:: http://img.shields.io/badge/docs-stable-yellow.svg
   :target: http://pythonhosted.org/bob.kaldi/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/latest/bob/bob.kaldi/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.kaldi/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.kaldi/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.kaldi
.. image:: http://img.shields.io/pypi/v/bob.kaldi.svg
   :target: https://pypi.python.org/pypi/bob.kaldi
.. image:: http://img.shields.io/pypi/dm/bob.kaldi.svg
   :target: https://pypi.python.org/pypi/bob.kaldi


===========================
 Python Bindings for Kaldi
===========================

This package provides pythonic bindings for Kaldi_ functionality so it can be
seamlessly integrated with Python-based workflows. It is a part fo the signal-
processing and machine learning toolbox Bob_.


Installation
------------

This package depends on both Bob_ and Kaldi_. To install Bob_ follow our
installation_ instructions. Kaldi_ is also bundled in our conda channnels which
means you can install Kaldi_ using conda easily too. After you have installed
Bob_, please follow these instructions to install Kaldi_ too.

  # BOB_ENVIRONMENT is the name of your conda enviroment.
  $ source activate BOB_ENVIRONMENT
  $ conda install kaldi
  $ pip install bob.kaldi


Documentation
-------------

For further documentation on this package, please read the `Stable Version
<http://pythonhosted.org/bob.kaldi/index.html>`_ or the `Latest Version
<https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.kaldi/master/index.html>`_
of the documentation.  For a list of tutorials on this or the other packages of
Bob_, or information on submitting issues, asking questions and starting
discussions, please visit its website.


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.

.. _bob: https://www.idiap.ch/software/bob
.. _kaldi: http://kaldi-asr.org/
.. _mailing list: https://www.idiap.ch/software/bob/discuss
.. _installation: https://www.idiap.ch/software/bob/install
