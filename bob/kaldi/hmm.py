#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Milos Cernak <milos.cernak@idiap.ch>
# Septebmer 9, 2017

import logging
import os
import tempfile
from subprocess import PIPE
from subprocess import Popen

from . import io

logger = logging.getLogger(__name__)


def train_mono(
    feats,
    trans_words,
    fst_L,
    topology_in,
    shared_phones="",
    numgauss=1000,
    power=0.25,
    num_iters=40,
    beam=6,
):
    """Monophone model training.

    Parameters
    ----------
    feats: dict
        The input cepstral features (2D array of 32-bit floats).
    trans_words: str
        Text transcription of the `feats` (the word labels)
    fst_L: str
        A filename of the lexicon compiled as FST.
    topology_in : str
        A topology file that specifies 3-state left-to-right HMM, and
        default transition probs.

    shared_phones : :obj:`str`, optional
         A filename of the of phones whose pdfs should be shared.
    numgauss : :obj:`int`, optional
        A number of Gaussians of GMMs.
    power : :obj:`float`, optional
        Power to allocate Gaussians to states.
    num_iters : :obj:`int`, optional
        A number of iteration for re-estimation of GMMs.
    beam : :obj:`float`, optional
        Decoding beam used in alignment.

    Returns
    -------
    str
        The mono-phones acoustic models.

    """

    feat_dim = -1
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ark") as arkf:
        with open(arkf.name, "wb") as f:
            for k in feats.keys():
                uttid = k
                io.write_mat(f, feats[k], key=uttid.encode("utf-8"))
                if feat_dim < 1:
                    (m, feat_dim) = feats[k].shape

    with tempfile.NamedTemporaryFile(delete=False, suffix=".top") as topof:
        with open(topof.name, "wt") as f:
            f.write(topology_in)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tra") as traf:
        with open(traf.name, "wt") as f:
            f.write(trans_words)

    binary1 = "gmm-init-mono"
    cmd1 = [binary1]
    binary2 = "compile-train-graphs"
    cmd2 = [binary2]
    binary3 = "align-equal-compiled"
    cmd3 = [binary3]
    binary4 = "gmm-acc-stats-ali"
    cmd4 = [binary4]
    binary5 = "gmm-est"
    cmd5 = [binary5]
    binary6 = "gmm-align-compiled"

    with tempfile.NamedTemporaryFile(
        suffix=".mdl"
    ) as initf, tempfile.NamedTemporaryFile(
        suffix=".tree"
    ) as treef, tempfile.NamedTemporaryFile(
        suffix=".fst"
    ) as fstf, tempfile.NamedTemporaryFile(
        suffix=".ali"
    ) as alif, tempfile.NamedTemporaryFile(
        suffix=".acc"
    ) as accf, tempfile.NamedTemporaryFile(
        suffix=".est"
    ) as estf:

        if shared_phones != "":
            cmd1 += [
                "--shared-phones=" + str(shared_phones),
            ]
        cmd1 += [
            "--train-feats=ark:copy-feats ark:" + arkf.name + " ark:-|",
            topof.name,
            str(feat_dim),
            initf.name,
            treef.name,
        ]
        with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
            pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe1.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd2 += [
            treef.name,
            initf.name,
            str(fst_L),
            "ark,t:" + traf.name,
            "ark,t:" + fstf.name,
        ]
        with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
            pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe2.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd3 += [
            "ark,t:" + fstf.name,
            "ark:" + arkf.name,
            "ark,t:" + alif.name,
        ]
        with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
            pipe3 = Popen(cmd3, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe3.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd4 += [
            initf.name,
            "ark:" + arkf.name,
            "ark,t:" + alif.name,
            accf.name,
        ]
        with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
            pipe4 = Popen(cmd4, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe4.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd5 += [
            "--min-gaussian-occupancy=3",
            "--mix-up=" + str(numgauss),
            "--power=" + str(power),
            "--binary=false",
            initf.name,
            accf.name,
            estf.name,
        ]
        with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
            pipe5 = Popen(cmd5, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe5.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        inModel = estf.name
        for x in range(0, num_iters):
            logger.info("Training pass " + str(x))

            cmd6 = [
                binary6,
                "--transition-scale=1.0",
                "--acoustic-scale=0.1",
                "--self-loop-scale=0.1",
                "--beam=" + str(beam),
                "--retry-beam=" + str(beam * 4),
                "--careful=false",
                inModel,
                "ark,t:" + fstf.name,
                "ark:" + arkf.name,
                "ark:" + alif.name,
            ]
            with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
                pipe6 = Popen(cmd6, stdin=PIPE, stdout=PIPE, stderr=logfile)
                pipe6.communicate()
                with open(logfile.name) as fp:
                    logtxt = fp.read()
                    logger.debug("%s", logtxt)

            cmd7 = [
                binary4,
                inModel,
                "ark:" + arkf.name,
                "ark:" + alif.name,
                accf.name,
            ]
            with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
                pipe7 = Popen(cmd7, stdin=PIPE, stdout=PIPE, stderr=logfile)
                pipe7.communicate()
                with open(logfile.name) as fp:
                    logtxt = fp.read()
                    logger.debug("%s", logtxt)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".est") as itf:
                cmd8 = [
                    binary5,
                    "--binary=false",
                    "--mix-up=" + str(numgauss),
                    "--power=" + str(power),
                    inModel,
                    accf.name,
                    itf.name,
                ]
                with tempfile.NamedTemporaryFile(suffix=".log") as logfile:
                    pipe8 = Popen(cmd8, stdin=PIPE, stdout=PIPE, stderr=logfile)
                    pipe8.communicate()
                    with open(logfile.name) as fp:
                        logtxt = fp.read()
                        logger.debug("%s", logtxt)

                if x > 0:  # do not remove estf.name; just itf.name
                    os.unlink(inModel)
                inModel = itf.name

    # shutil.copyfile(inModel,'final.mdl')
    os.unlink(arkf.name)
    os.unlink(topof.name)
    os.unlink(traf.name)

    with open(inModel) as fp:
        hmmtxt = fp.read()
        return hmmtxt
