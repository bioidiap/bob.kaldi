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
seemlessly integrated with Python-based workflows. It is a part fo the signal-processing and machine learning toolbox
Bob_.


Installation
------------

Follow our `installation`_ instructions. Then, using the Python interpreter
provided by the distribution, build this package with::

  $ buildout

To be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies
<https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

This package also requires that Kaldi_ is properly installed alongside the
Python interpreter you're using, under the directory ``<PREFIX>/lib/kaldi``,
along with all necessary scripts and compiled binaries.


Documentation
-------------

For further documentation on this package, please read the `Stable Version
<http://pythonhosted.org/bob.kaldi/index.html>`_ or the `Latest Version
<https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.kaldi/master/index.html>`_
of the documentation.  For a list of tutorials on this or the other packages ob
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