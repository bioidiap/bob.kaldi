#!/usr/bin/env python
#
# Milos Cernak <milos.cernak@idiap.ch>
# March 1, 2017
#

import os

from . import io

from subprocess import PIPE, Popen
import tempfile
import shutil

import logging
logger = logging.getLogger(__name__)


def ubm_train(feats, ubmname, num_threads=4, num_frames=500000,
              min_gaussian_weight=0.0001, num_gauss=2048, num_gauss_init=0,
              num_gselect=30, num_iters_init=20, num_iters=4,
              remove_low_count_gaussians=True):
    """Implements Kaldi egs/sre10/v1/train_diag_ubm.sh

    Parameters
    ----------
    feats : numpy.ndarray
            A 2D numpy ndarray object containing MFCCs.

    ubmname : str
            A path to the UBM model.

    num_threads : int, optional
            Number of threads used for statistics accumulation.
    num_frames : int, optional
            Number of feature vectors to store in memory and train on
            (randomly chosen from the input features).
    min_gaussian_weight : float, optional
            Kaldi MleDiagGmmOptions: Min Gaussian weight before we
            remove it.
    num_gauss : int, optional
            Number of Gaussians in the model.
    num_gauss_init : int, optional
            Number of Gaussians in the model initially (if nonzero and
            less than num_gauss, we'll do mixture splitting).
    num_gselect : int, optional
            Number of Gaussians to keep per frame.
    num_iters_init : int, optional
            Number of iterations of training for initialization of the
            single diagonal GMM.
    num_iters : int, optional
            Number of iterations of training.
    remove_low_count_gaussians : bool, optional
            Kaldi MleDiagGmmOptions: If true, remove Gaussians that
            fall below the floors.

    Returns
    -------
    str
            A path to the the trained ubm model.

    """

    binary1 = 'gmm-global-init-from-feats'
    binary2 = 'subsample-feats'
    binary3 = 'gmm-gselect'
    binary4 = 'gmm-global-acc-stats'
    binary5 = 'gmm-global-est'

    # 1. Initialize a single diagonal GMM
    cmd1 = [binary1] # gmm-global-init-from-feats
    cmd1 += [
        '--num-threads=' + str(num_threads),
        '--num-frames=' + str(num_frames),
        '--min-gaussian-weight=' + str(min_gaussian_weight),
        '--num-gauss=' + str(num_gauss),
        '--num-gauss-init=' + str(num_gauss_init),
        '--num-iters=' + str(num_iters_init),
    ]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dubm') as \
    initfile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd1 += [
            'ark:-',
            initfile.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        # write ark file into pipe.stdin
        io.write_mat(pipe1.stdin, feats, key=b'abc')
        # pipe1.stdin.close()
        pipe1.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)
            
    # 2. Store Gaussian selection indices on disk-- this speeds up the
    # training passes.
    # subsample-feats --n=$subsample ark:- ark:- |"
    cmd = [binary2] # subsample-feats
    with tempfile.NamedTemporaryFile(suffix='.ark') as arkfile, \
    tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as gselfile:
        cmd += [
            '--n=5',
            'ark:-',
            'ark:' + arkfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=logfile)
            io.write_mat(pipe.stdin, feats, key=b'abc')
            pipe.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)
        cmd2 = [binary3] # gmm-gselect
        cmd2 += [
            '--n=' + str(num_gselect),
            initfile.name,
            'ark:' + arkfile.name,
            'ark:|gzip -c >' + gselfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
            # write ark file into pipe.stdin
            # io.write_mat(pipe2.stdin, feats, key='abc')
            # pipe2.stdin.close()
            pipe2.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)
                
        inModel = initfile.name
        for x in range(0, num_iters):
            logger.info("Training pass " + str(x))
            # Accumulate stats.
            with tempfile.NamedTemporaryFile(suffix='.acc') as accfile:
                cmd3 = [binary4] # gmm-global-acc-stats
                cmd3 += [
                    '--gselect=ark,s,cs:gunzip -c ' + gselfile.name + '|',
                    inModel,
                    'ark:' + arkfile.name,
                    accfile.name,
                ]
                with tempfile.NamedTemporaryFile(
                        suffix='.log') as logfile:
                    pipe3 = Popen(cmd3, stdin=PIPE,
                                  stdout=PIPE, stderr=logfile)
                    # write ark file into pipe.stdin
                    # io.write_mat(pipe3.stdin, feats, key='abc')
                    # pipe3.stdin.close()
                    pipe3.communicate()
                    with open(logfile.name) as fp:
                        logtxt = fp.read()
                        logger.debug("%s", logtxt)
                # Don't remove low-count Gaussians till last iter.
                if x < num_iters - 1:
                    opt = '--remove-low-count-gaussians=false'
                else:
                    opt = '--remove-low-count-gaussians=true'
                cmd4 = [binary5] # gmm-global-est
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.dump') as estfile:
                    cmd4 += [
                        opt,
                        '--min-gaussian-weight=' + str(min_gaussian_weight),
                        inModel,
                        accfile.name,
                        estfile.name,
                    ]
                    with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
                        pipe4 = Popen(cmd4, stdin=PIPE,
                                      stdout=PIPE, stderr=logfile)
                        pipe4.communicate()
                        with open(logfile.name) as fp:
                            logtxt = fp.read()
                            logger.debug("%s", logtxt)
                            
                    os.unlink(inModel)
                    inModel = estfile.name
                    
    os.unlink(gselfile.name)
    shutil.copyfile(estfile.name, ubmname)
    shutil.copyfile(estfile.name, ubmname + '.dubm')
    os.unlink(estfile.name)
    
    return ubmname + '.dubm'


def ubm_full_train(feats, dubmname, num_gselect=20, num_iters=4,
                   min_gaussian_weight=1.0e-04):
    """ Implements Kaldi egs/sre10/v1/train_full_ubm.sh

    Parameters
    ----------
    feats : numpy.ndarray
            A 2D numpy ndarray object containing MFCCs.

    dubmname : str
            A path to the UBM model.

    num_gselect : int, optional
            Number of Gaussians to keep per frame.
    num_iters : int, optional
            Number of iterations of training.
    min_gaussian_weight : float, optional
            Kaldi MleDiagGmmOptions: Min Gaussian weight before we
            remove it.

    Returns
    -------
    str
            A path to the the trained full covariance UBM model.

    """

    binary1 = 'gmm-global-to-fgmm'
    binary2 = 'fgmm-global-to-gmm'
    binary3 = 'subsample-feats'
    binary4 = 'gmm-gselect'
    binary5 = 'fgmm-global-acc-stats'
    binary6 = 'fgmm-global-est'

    origdubm = dubmname
    dubmname += '.dubm'

    # 1. Init (diagonal GMM to full-cov. GMM)
    # gmm-global-to-fgmm $srcdir/final.dubm $dir/0.ubm || exit 1;
    cmd1 = [binary1] # gmm-global-to-fgmm
    inModel = ''
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ubm') as \
    initfile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        inModel = initfile.name
        cmd1 += [
            dubmname,
            inModel,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe1.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)
            
    # 2. doing Gaussian selection (using diagonal form of model; \
    # selecting $num_gselect indices)
    # gmm-gselect --n=$num_gselect "fgmm-global-to-gmm $dir/0.ubm - \
    # |" "$feats" \
    #   "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
    cmd2 = [binary2] # fgmm-global-to-gmm
    with tempfile.NamedTemporaryFile(suffix='.dubm') as dubmfile, \
         tempfile.NamedTemporaryFile(suffix='.ark') as arkfile, \
         tempfile.NamedTemporaryFile(suffix='.gz') as gselfile:
        cmd2 += [
            inModel,
            dubmfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe2.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)
        # subsample-feats --n=$subsample ark:- ark:- |"
        cmd = [binary3] # subsample-feats
        cmd += [
            '--n=5',
            'ark:-',
            'ark:' + arkfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=logfile)
            io.write_mat(pipe.stdin, feats, key=b'abc')
            pipe.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)
        cmd3 = [binary4] # gmm-gselect
        cmd3 += [
            '--n=' + str(num_gselect),
            dubmfile.name,
            'ark:' + arkfile.name,
            'ark:|gzip -c >' + gselfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe3 = Popen(cmd3, stdin=PIPE,
                          stdout=PIPE, stderr=logfile)
            # io.write_mat(pipe3.stdin, feats, key='abc')
            # pipe3.stdin.close()
            pipe3.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)
        # 3 est num_iters times
        for x in range(0, num_iters):
            logger.info("Training pass " + str(x))
            # Accumulate stats.
            with tempfile.NamedTemporaryFile(
                    suffix='.acc') as accfile:
                cmd4 = [binary5] # fgmm-global-acc-stats
                cmd4 += [
                    '--gselect=ark,s,cs:gunzip -c ' +
                    gselfile.name + '|',
                    inModel,
                    'ark:' + arkfile.name,
                    accfile.name,
                ]
                with tempfile.NamedTemporaryFile(
                        suffix='.log') as logfile:
                    pipe4 = Popen(cmd4, stdin=PIPE,
                                  stdout=PIPE, stderr=logfile)
                    # io.write_mat(pipe4.stdin, feats, key='abc')
                    pipe4.communicate()
                    with open(logfile.name) as fp:
                        logtxt = fp.read()
                        logger.debug("%s", logtxt)
                # Don't remove low-count Gaussians till last iter.
                if x < num_iters - 1:
                    opt = '--remove-low-count-gaussians=false'
                else:
                    opt = '--remove-low-count-gaussians=true'
                cmd5 = [binary6] # fgmm-global-est
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.dump') as estfile:
                    cmd5 += [
                        opt,
                        '--min-gaussian-weight=' +
                        str(min_gaussian_weight),
                        inModel,
                        accfile.name,
                        estfile.name,
                    ]
                    with tempfile.NamedTemporaryFile(
                            suffix='.log') as logfile:
                        pipe5 = Popen(cmd5, stdin=PIPE,
                                      stdout=PIPE, stderr=logfile)
                        pipe5.communicate()
                        with open(logfile.name) as fp:
                            logtxt = fp.read()
                            logger.debug("%s", logtxt)
                        os.unlink(inModel)
                        inModel = estfile.name
                        
    shutil.copyfile(estfile.name, origdubm + '.fubm')
    os.unlink(estfile.name)
    
    return origdubm + '.fubm'

def ubm_enroll(feats, ubm_file):
    """Performes MAP adaptation of GMM-UBM model.

    Parameters
    ----------
    feats : numpy.ndarray
        A 2D numpy ndarray object containing MFCCs.
    ubm_file : str
        A path to the Kaldi global GMM.


    Returns
    -------
    str
        A path to the enrolled GMM.

    """

    binary1 = 'gmm-global-acc-stats'
    binary2 = 'global-gmm-adapt-map'

    # 1. Accumulate stats for training a diagonal-covariance GMM.
    cmd1 = [binary1] # gmm-global-acc-stats
    cmd1 += [
        ubm_file,
        'ark:-',
        '-',
    ]
    cmd2 = [binary2] # global-gmm-adapt-map
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ubm') as \
    estfile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd2 += [
            '--update-flags=m',
            ubm_file,
            '-',
            estfile.name,
        ]

        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe2 = Popen(cmd2, stdin=pipe1.stdout,
                      stdout=PIPE, stderr=logfile)
        # write ark file into pipe1.stdin
        io.write_mat(pipe1.stdin, feats, key=b'abc')
        pipe1.stdin.close()
        pipe2.communicate()
        
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

    return estfile.name

def gmm_score(feats, gmm_file, ubm_file):
    """Print out per-frame log-likelihoods for input utterance.

    Parameters
    ----------
    feats : numpy.ndarray
        A 2D numpy ndarray object containing MFCCs.

    gmm_file : str
        A path to Kaldi adapted global GMM.
    ubm_file : str
        A path to Kaldi global GMM.


    Returns
    -------
    float
        The average of per-frame log-likelihoods.

    """

    binary1 = 'gmm-global-get-frame-likes'
    
    models = [
        gmm_file,
        ubm_file
    ]
    ret = [0, 0]
    # import ipdb; ipdb.set_trace()
    for i, m in enumerate(models):
        # 1. Accumulate stats for training a diagonal-covariance GMM.
        cmd1 = [binary1]
        cmd1 += [
            '--average=true',
            m,
            'ark:-',
            'ark,t:-',
        ]

        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
            # write ark file into pipe1.stdin
            io.write_mat(pipe1.stdin, feats, key=b'abc')
            pipe1.stdin.close()
            # pipe1.communicate()

            # read ark from pipe1.stdout
            rettxt = pipe1.stdout.readline()
            ret[i] = float(rettxt.split()[1])
            # ret = [mat for name,mat in io.read_mat_ark(pipe1.stdout)][0]

            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

    return ret[0] - ret[1]

# def gmm_score_fast(feats, gmm_file, ubm_file):
#   """Print out per-frame log-likelihoods for each utterance, as an archive
#   of vectors of floats.  If --average=true, prints out the average per-frame
#   log-likelihood for each utterance, as a single float.

#   Parameters:

#     feats (numpy.ndarray): A 2D numpy ndarray object containing MFCCs.
#     ubm_file (string)    : A Kaldi (spk.-dep.) global GMM.


#   Returns:

#     float: The score.


#   Raises:

#     RuntimeError: if any problem was detected during the conversion.

#     IOError: if the binary to be executed does not exist

#   """

# # gmm-gselect --n=10  \
# #     ubm.gmm \
# #     "$feats" \
# #     ark:gselect.ark

# # gmm-compute-likes-gmmubm \
# # --average=true --gselect="ark:gselect.ark"
# # adapted.gmm \
# # ubm.gmm \
# # "$feats" ark,t:score.ark

#   ret=0
#   with tempfile.NamedTemporaryFile(suffix='.ark') as gselfile:
#     # 1. Accumulate stats for training a diagonal-covariance GMM.
#     binary1 = 'gmm-gselect'
#     cmd1 = [binary1]
#     cmd1 += [
#       '--n=10',
#       ubm_file,
#       'ark:-',
#       'ark:' + gselfile.name,
#     ]
#     with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
#       pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
#       # write ark file into pipe1.stdin
#       io.write_mat(pipe1.stdin, feats, key='abc')
#       pipe1.stdin.close()
#       # pipe1.communicate()

#       with open(logfile.name) as fp:
#         logtxt = fp.read()
#         logger.debug("%s", logtxt)

#     binary2 = 'gmm-compute-likes-gmmubm'
#     cmd2 = [binary2]
#     cmd2 += [
#       '--average=true',
#       '--gselect=ark:' + gselfile.name,
#       gmm_file,
#       ubm_file,
#       'ark:-',
#       'ark,t:-',
#     ]
#     with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
#       pipe2 = Popen (cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
#       # write ark file into pipe1.stdin
#       io.write_mat(pipe2.stdin, feats, key='abc')
#       pipe2.stdin.close()
#       # pipe2.communicate()

#       with open(logfile.name) as fp:
#         logtxt = fp.read()
#         logger.debug("%s", logtxt)

#       # read ark from pipe1.stdout
#       # import ipdb; ipdb.set_trace()
#       rettxt = pipe2.stdout.readline().split()
#       if len(rettxt) > 1:
#         ret = rettxt[1]
#       else:
#         logger.debug("ERROR in gmm_score_fast; outputting score 0")

#   return float(ret)
