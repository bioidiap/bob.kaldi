#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marc Ferras Font <marc.ferras@idiap.ch>
# Mon 11 Jul 2016 10:39:15 CEST

"""Various utilities for Kaldi I/O"""

from sys import executable
from os.path import dirname, join


def kaldi_path(suffix=None):
  '''Returns the installation root prefix for Kaldi binaries and scripts

  Parameters:

    path (list, Optional): A path to be suffixed to the kaldi root prefix,
      composed by parts in a list

  '''

  prefix = join(dirname(dirname(executable)), 'lib', 'kaldi')

  return prefix if suffix is None else join(prefix, *suffix)
