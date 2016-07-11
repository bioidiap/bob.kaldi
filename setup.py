#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 11 Jul 2016 10:20:55 CEST

"""A package that contains helpers for Kaldi tool usage in Python environments
"""

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements
install_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

setup(

    name="bob.kaldi",
    version=version,
    description="Kaldi Python bindings for BEAT",
    url='http://gitlab.idiap.ch/biometric/bob.kaldi',
    license="GPLv3",
    author='Marc Ferras Font',
    author_email='marc.ferras@idiap.ch',
    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=install_requires,

    entry_points = {
      'console_scripts': [
        #'bob_kaldi_mfcc.py = bob.kaldi.scripts:mfcc',
      ],
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

)
