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


def ivector_train(feats, fubm, ivector_extractor, num_gselect=20,
                  ivector_dim=600, use_weights=False, num_iters=5,
                  min_post=0.025, num_samples_for_weights=3,
                  posterior_scale=1.0):
    """Implements Kaldi egs/sre10/v1/train_ivector_extractor.sh

    Parameters
    ----------
    feats : numpy.ndarray
        A 2D numpy ndarray object containing MFCCs.
    fubm : str
        A path to full-diagonal UBM file
    ivector_extractor : str
        A path to the ivector extractor

    num_gselect : :obj:`int`, optional
        Number of Gaussians to keep per frame.
    ivector_dim : :obj:`int`, optional
        Dimension of iVector.
    use_weights : :obj:`bool`, optional
        If true, regress the log-weights on the iVector
    num_iters : :obj:`int`, optional
        Number of iterations of training.
    min_post : :obj:`float`, optional
        If nonzero, posteriors below this threshold will be pruned
        away and the rest will be renormalized to sum to one.
    num_samples_for_weights : :obj:`int`, optional
        Number of samples from iVector distribution to use for
        accumulating stats for weight update.  Must be >1.
    posterior_scale : :obj:`float`, optional
        A posterior scaling with a global scale.

    Returns
    -------
    str
        A path to the iVector extractor.

    """

    binary1 = 'fgmm-global-to-gmm'
    binary2 = 'ivector-extractor-init'
    binary3 = 'gmm-gselect'
    binary4 = 'fgmm-global-gselect-to-post'
    binary5 = 'scale-post'
    binary6 = 'ivector-extractor-acc-stats'
    binary7 = 'ivector-extractor-est'

    # 1. Create Kaldi training data structure
    # ToDo: implement Bob's function for that
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ark') as arkfile:
        with open(arkfile.name, 'wb') as f:
            if feats.ndim == 3:
                for i, utt in enumerate(feats):
                    uttid = 'utt' + str(i)
                    io.write_mat(f, utt, key=uttid.encode('utf-8'))
            else:
                io.write_mat(f, feats, key=b'utt0')

    # Initialize the i-vector extractor using the FGMM input
    cmd1 = [binary1] # fgmm-global-to-gmm
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dubm') as \
    dubmfile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd1 += [
            fubm,
            dubmfile.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe1.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

    cmd2 = [binary2] # ivector-extractor-init
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ie') as \
    iefile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd2 += [
            '--ivector-dim=' + str(ivector_dim),
            '--use-weights=' + str(use_weights).lower(),
            fubm,
            iefile.name,
        ]
        pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe2.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)
            
        inModel = iefile.name  # for later re-estimation

    # Do Gaussian selection and posterior extracion
    # gmm-gselect --n=$num_gselect $dir/final.dubm "$feats" ark:- \| \
    # fgmm-global-gselect-to-post --min-post=$min_post $dir/final.ubm \
    # "$feats" ark,s,cs:-  ark:- \| \
    # scale-post ark:- $posterior_scale "ark:|gzip -c >$dir/post.JOB.gz"
    cmd3 = [binary3] # gmm-gselect
    with tempfile.NamedTemporaryFile(suffix='.gsel') as gselfile, \
        tempfile.NamedTemporaryFile(suffix='.post.gz') as postfile:
        cmd3 += [
            '--n=' + str(num_gselect),
            dubmfile.name,
            'ark:' + arkfile.name,
            'ark:' + gselfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe3 = Popen(cmd3, stdin=PIPE, stdout=PIPE, stderr=logfile)
            # io.write_mat(pipe3.stdin, feats, key='abc')
            pipe3.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd4 = [binary4] # fgmm-global-gselect-to-post
        cmd4 += [
            '--min-post=' + str(min_post),
            fubm,
            'ark:' + arkfile.name,
            'ark:' + gselfile.name,
            'ark:-',
        ]
        # 'ark,s,cs:' + gselfile.name,
        cmd5 = [binary5] # scale-post
        cmd5 += [
            'ark:-',
            str(posterior_scale),
            'ark:|gzip -c >' + postfile.name,
        ]

        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe4 = Popen(cmd4, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe5 = Popen(cmd5, stdin=pipe4.stdout,
                          stdout=PIPE, stderr=logfile)
            # io.write_mat(pipe4.stdin, feats, key='abc')
            # pipe4.stdin.close()
            pipe5.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        # Estimate num_iters times
        for x in range(0, num_iters):
            logger.info("Training pass " + str(x))
            # Accumulate stats.
            with tempfile.NamedTemporaryFile(suffix='.acc') as accfile:
                cmd6 = [binary6] # ivector-extractor-acc-stats
                cmd6 += [
                    '--num-threads=4',
                    '--num-samples-for-weights=' +
                    str(num_samples_for_weights),
                    inModel,
                    'ark:' + arkfile.name,
                    'ark:gunzip -c ' + postfile.name + '|',
                    accfile.name,
                ]
                # ark,s,cs

                with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
                    pipe6 = Popen(cmd6, stdin=PIPE,
                                  stdout=PIPE, stderr=logfile)
                    # io.write_mat(pipe6.stdin, feats, key='abc')
                    pipe6.communicate()
                    with open(logfile.name) as fp:
                        logtxt = fp.read()
                        logger.debug("%s", logtxt)

                cmd7 = [binary7] # ivector-extractor-est
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix='.ie') as estfile, \
                     tempfile.NamedTemporaryFile(suffix='.log') as \
                         logfile:
                    cmd7 += [
                        '--num-threads=4',
                        inModel,
                        accfile.name,
                        estfile.name,
                    ]
                    pipe7 = Popen(cmd7, stdin=PIPE,
                                  stdout=PIPE, stderr=logfile)
                    pipe7.communicate()
                    with open(logfile.name) as fp:
                        logtxt = fp.read()
                        logger.debug("%s", logtxt)

                    os.unlink(inModel)
                    inModel = estfile.name

    shutil.copyfile(inModel, ivector_extractor)
    os.unlink(inModel)

    return ivector_extractor # ToDo: covert to the string


def ivector_extract(feats, fubm, ivector_extractor, num_gselect=20,
                    min_post=0.025, posterior_scale=1.0):
    """Implements Kaldi egs/sre10/v1/extract_ivectors.sh

    Parameters
    ----------
    feats : numpy.ndarray
        A 2D numpy ndarray object containing MFCCs.
    fubm : str
        A path to full-diagonal UBM file
    ivector_extractor : str
        A path to global GMM file.
    num_gselect : :obj:`int`, optional
        Number of Gaussians to keep per frame.
    min_post : :obj:`float`, optional
        If nonzero, posteriors below this threshold will be pruned
        away and the rest will be renormalized to sum to one.
    posterior_scale : :obj:`float`, optional
        A posterior scaling with a global scale.

    Returns
    -------
    numpy.ndarray
        The iVectors calculated for the input signal.

    """

    binary1 = 'fgmm-global-to-gmm'
    binary2 = 'gmm-gselect'
    binary3 = 'fgmm-global-gselect-to-post'
    binary4 = 'scale-post'
    binary5 = 'ivector-extract'

    # import ipdb; ipdb.set_trace()
    # ivector-extract --verbose=2 $srcdir/final.ie "$feats" ark,s,cs:- \
    # ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;

    # Initialize the i-vector extractor using the FGMM input
    cmd1 = [binary1] # fgmm-global-to-gmm
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dubm') as \
    dubmfile, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd1 += [
            fubm,
            dubmfile.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe1.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)
        
    cmd = [binary2] # gmm-gselect
    with tempfile.NamedTemporaryFile(suffix='.gsel') as gselfile, \
    tempfile.NamedTemporaryFile(suffix='.post') as postfile:
        cmd += [
            '--n=' + str(num_gselect),
            dubmfile.name,
            'ark:-',
            'ark:' + gselfile.name,
        ]
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=logfile)
            io.write_mat(pipe.stdin, feats, key=b'abc')
            pipe.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd2 = [binary3] # fgmm-global-gselect-to-post
        cmd2 += [
            '--min-post=' + str(min_post),
            fubm,
            'ark:-',
            'ark,s,cs:' + gselfile.name,
            'ark:-',
        ]
        cmd3 = [binary4] # scale-post
        cmd3 += [
            'ark:-',
            str(posterior_scale),
            'ark:' + postfile.name,
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
            pipe3 = Popen(cmd3, stdin=pipe2.stdout,
                          stdout=PIPE, stderr=logfile)
            io.write_mat(pipe2.stdin, feats, key=b'abc')
            pipe2.stdin.close()
            pipe3.communicate()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

        cmd4 = [binary5] # ivector-extract
        cmd4 += [
            ivector_extractor,
            'ark:-',
            'ark,s,cs:' + postfile.name,
            'ark:-',
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
            pipe4 = Popen(cmd4, stdin=PIPE, stdout=PIPE, stderr=logfile)
            io.write_mat(pipe4.stdin, feats, key=b'abc')
            pipe4.stdin.close()
            with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

            # read ark from pipe1.stdout
            ret = [mat for name, mat in io.read_vec_flt_ark(
                pipe4.stdout)][0]
            return ret


def plda_train(feats, enroller_file):
    """Implements Kaldi egs/sre10/v1/plda_scoring.sh

    Parameters
    ----------

    feats : numpy.ndarray
        A 2D numpy ndarray object containing MFCCs.

    enroller_file : str
        A path to adapted GMM file

    Returns
    -------
    str
        A path to trained PLDA model.

    """
# ivector-compute-plda ark:$plda_data_dir/spk2utt \
# "ark:ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp  ark:- |" \
# $plda_ivec_dir/plda || exit 1;

    binary1 = 'ivector-normalize-length'
    binary2 = 'ivector-compute-plda'
    binary3 = 'ivector-mean'

    logger.debug("-> PLDA calculation")
    # 1. Create Kaldi training data structure
    # import ipdb; ipdb.set_trace()
    with tempfile.NamedTemporaryFile(
            mode='w+t', suffix='.spk2utt', delete=False) as spkfile, \
        tempfile.NamedTemporaryFile(
                delete=False, suffix='.ark') as arkfile, \
        open(arkfile.name, 'wb') as f:
        
        i = 0
        for spk in feats:
            j = 0
            spkid = 'spk' + str(i)
            spkfile.write(spkid)
            for utt in spk:
                # print i, j
                spkutt = spkid + 'utt' + str(j)
                io.write_vec_flt(f, utt, key=spkutt.encode('utf-8'))
                spkfile.write(' ' + spkutt)
                j += 1
            spkfile.write("\n")
            i += 1

    cmd1 = [binary1] # ivector-normalize-length
    cmd1 += [
        'ark:' + arkfile.name,
        'ark:-',
    ]
    cmd2 = [binary2] # ivector-compute-plda

    with tempfile.NamedTemporaryFile(suffix='.plda') as pldafile, \
        tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd2 += [
            'ark,t:' + spkfile.name,
            'ark:-',
            pldafile.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe2 = Popen(cmd2, stdin=pipe1.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe2.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

        shutil.copyfile(pldafile.name, enroller_file + '.plda')

    # compute global mean
    # ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
    # ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;
    # import ipdb; ipdb.set_trace()
    cmd3 = [binary3] # ivector-mean
    with tempfile.NamedTemporaryFile(suffix='.mean') as meanfile, \
        tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd3 += [
            'ark:-',
            meanfile.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe3 = Popen(cmd3, stdin=pipe1.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe3.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

        shutil.copyfile(meanfile.name, enroller_file + '.plda.mean')

    os.unlink(spkfile.name)
    os.unlink(arkfile.name)

    return enroller_file + '.plda'


def plda_enroll(feats, enroller_file):
    """Implements Kaldi egs/sre10/v1/plda_scoring.sh

    Parameters
    ----------

    feats : numpy.ndarray
        A 2D numpy ndarray object containing iVectors.

    enroller_file : str
        A path to enrolled/adapted GMM file.

    Returns
    -------
    str
        A path to enrolled PLDA model.

    """

    binary1 = 'ivector-normalize-length'
    binary2 = 'ivector-mean'
    binary3 = 'ivector-normalize-length'
    binary4 = 'ivector-subtract-global-mean'
    binary5 = 'ivector-normalize-length'
    
    # ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
    # ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
    # ivector-normalize-length ark:-
    # ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector
    logger.debug("-> PLDA enrollment")
    # 1. Create Kaldi training data structure
    # ToDO: change in future
    with tempfile.NamedTemporaryFile(
            mode='w+t', suffix='.spk2utt', delete=False) as spkfile, \
        tempfile.NamedTemporaryFile(
                delete=False, suffix='.ark') as arkfile, \
        open(arkfile.name, 'wb') as f:
            i = 0
            j = 0
            spkid = 'spk' + str(i)
            spkfile.write(spkid)
            for utt in feats:
                # print i, j
                spkutt = spkid + 'utt' + str(j)
                io.write_vec_flt(f, utt, key=spkutt.encode('utf-8'))
                spkfile.write(' ' + spkutt)
                j += 1
            spkfile.write("\n")
            i += 1

    cmd1 = [binary1] # ivector-normalize-length
    cmd1 += [
        'ark:' + arkfile.name,
        'ark:-',
    ]
    cmd2 = [binary2] # ivector-mean
    cmd2 += [
        'ark,t:' + spkfile.name,
        'ark:-',
        'ark:-',
    ]
    cmd3 = [binary3] # ivector-normalize-length
    cmd3 += [
        'ark:-',
        'ark:-',
    ]
    cmd4 = [binary4] # ivector-subtract-global-mean
    cmd4 += [
        enroller_file + '.plda.mean',
        'ark:-',
        'ark:-',
    ]
    cmd5 = [binary5] # ivector-normalize-length
    with tempfile.NamedTemporaryFile(
            delete=False, suffix='.ark') as spkarkfile, \
        tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        
        cmd5 += [
            'ark:-',
            'ark:' + spkarkfile.name,
        ]

        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe2 = Popen(cmd2, stdin=pipe1.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe3 = Popen(cmd3, stdin=pipe2.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe4 = Popen(cmd4, stdin=pipe3.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe5 = Popen(cmd5, stdin=pipe4.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe5.communicate()
        logger.debug("PLDA enrollment DONE ->")
        # with open(logfile.name) as fp:
        #   logtxt = fp.read()
        #   logger.debug("%s", logtxt)

    os.unlink(spkfile.name)
    os.unlink(arkfile.name)

    return spkarkfile.name


def plda_score(feats, model, ubm):
    """Implements Kaldi egs/sre10/v1/plda_scoring.sh

    Parameters
    ----------
    feats : numpy.ndarray
        A 2D numpy ndarray object containing iVectors.

    model : str
        A path to enrolled/adapted PLDA model.
    ubm : str
        A path to the PLDA model.

    Returns
    -------
    float
        A score.
    """
    # import ipdb; ipdb.set_trace()
    # ivector-plda-scoring --normalize-length=true \
    # --num-utts=ark:${enroll_ivec_dir}/num_utts.ark \
    # "ivector-copy-plda --smoothing=0.0 ${plda_ivec_dir}/plda - |" \
    # "ark:ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec \
    # scp:${enroll_ivec_dir}/spk_ivector.scp ark:- | \
    # ivector-normalize-length ark:- ark:- |" \
    # "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- | \
    # ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec ark:- ark:- | \
    # ivector-normalize-length ark:- ark:- |" \
    # "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || \
    # exit 1;

    logger.debug("-> PLDA scoring")
    # 1.

    binary1 = 'ivector-copy-plda'
    binary2 = 'ivector-normalize-length'
    binary3 = 'ivector-subtract-global-mean'
    binary4 = 'ivector-normalize-length'
    binary5 = 'ivector-plda-scoring'
    
    cmd1 = [binary1] # ivector-copy-plda

    # tests/probes
    cmd2 = [binary2] # ivector-normalize-length
    cmd2 += [
        'ark:-',
        'ark:-',
    ]

    cmd2 = [binary3] # ivector-subtract-global-mean
    cmd2 += [
        ubm + '.plda.mean',
        'ark:-',
        'ark:-',
    ]
    cmd3 = [binary4] # ivector-normalize-length
    cmd3 += [
        'ark:-',
        'ark:-',
    ]

    # scoring
    cmd4 = [binary5] # ivector-plda-scoring
    with tempfile.NamedTemporaryFile(
            mode='w+t', suffix='.trials', delete=False) as trials:
        trials.write("spk0 spk1\n")

    ret = 0

    # plda smooting
    with tempfile.NamedTemporaryFile(delete=False, suffix='.plda') as \
        plda, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd1 += [
            '--smoothing=0.0',
            ubm + '.plda',
            plda.name,
        ]
        pipe1 = Popen(cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe1.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.score') as \
        score, tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        cmd4 += [
            '--normalize-length=true',
            plda.name,
            'ark:' + model,
            'ark:-',
            trials.name,
            score.name,
        ]

        pipe2 = Popen(cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe3 = Popen(cmd3, stdin=pipe2.stdout,
                      stdout=PIPE, stderr=logfile)
        pipe4 = Popen(cmd4, stdin=pipe3.stdout,
                      stdout=PIPE, stderr=logfile)
        io.write_vec_flt(pipe2.stdin, feats, key=b'spk1')
        pipe2.stdin.close()
        pipe4.communicate()
        with open(logfile.name) as fp:
            logtxt = fp.read()
            logger.debug("%s", logtxt)

        with open(score.name) as fp:
            scoretxt = fp.readline()
            ret = float(scoretxt.split()[2])

    os.unlink(plda.name)
    os.unlink(trials.name)
    os.unlink(score.name)

    return ret
