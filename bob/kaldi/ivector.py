#!/usr/bin/env python
#
# Milos Cernak <milos.cernak@idiap.ch>
# March 1, 2017
# 

import os

import numpy as np

from . import io

from subprocess import PIPE, Popen
from os.path import join
import tempfile
import shutil

import logging
logger = logging.getLogger("bob.kaldi")

def ivector_train(feats, projector_file, num_gselect=20, ivector_dim=600, use_weights=False, num_iters=5, min_post=0.025, num_samples_for_weights=3, posterior_scale=1.0):
  """ Implements egs/sre10/v1/train_ivector_extractor.sh
  """
  fgmm_model=projector_file+'.fubm'

  # 1. Create Kaldi training data structure
  with tempfile.NamedTemporaryFile(delete=False, suffix='.ark') as arkfile:
    with open(arkfile.name,'wb') as f:
      if feats.ndim == 3:
        for i,utt in enumerate(feats):
          uttid='utt'+str(i)
          io.write_mat(f, utt, key=uttid)
      else:
        io.write_mat(f, feats, key='utt0')
          
  # Initialize the i-vector extractor using the FGMM input
  binary1 = 'fgmm-global-to-gmm'
  cmd1 = [binary1]
  with tempfile.NamedTemporaryFile(delete=False, suffix='.dubm') as dubmfile:
    cmd1 += [
      fgmm_model,
      dubmfile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe1.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

  binary2 = 'ivector-extractor-init'
  cmd2 = [binary2]
  with tempfile.NamedTemporaryFile(delete=False, suffix='.ie') as iefile:
    cmd2 += [
      '--ivector-dim=' + str(ivector_dim),
      '--use-weights=' + str(use_weights).lower(),
      fgmm_model,
      iefile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe2 = Popen (cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe2.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)
    inModel=iefile.name # for later re-estimation

  # Do Gaussian selection and posterior extracion
  # gmm-gselect --n=$num_gselect $dir/final.dubm "$feats" ark:- \| \
  # fgmm-global-gselect-to-post --min-post=$min_post $dir/final.ubm "$feats" \
  # ark,s,cs:-  ark:- \| \
  # scale-post ark:- $posterior_scale "ark:|gzip -c >$dir/post.JOB.gz"
  binary3 = 'gmm-gselect'
  cmd3 = [binary3]
  with tempfile.NamedTemporaryFile(suffix='.gsel') as gselfile:
    cmd3 += [
      '--n=' + str(num_gselect),
      dubmfile.name,
      'ark:' + arkfile.name,
      'ark:' + gselfile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe3 = Popen (cmd3, stdin=PIPE, stdout=PIPE, stderr=logfile)
      # io.write_mat(pipe3.stdin, feats, key='abc')
      pipe3.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

    with tempfile.NamedTemporaryFile(suffix='.post.gz') as postfile:
      binary4 = 'fgmm-global-gselect-to-post'
      cmd4 = [binary4]
      cmd4 += [
        '--min-post=' + str(min_post),
        fgmm_model,
        'ark:' + arkfile.name,
        'ark:' + gselfile.name,
        'ark:-',
      ]
      # 'ark,s,cs:' + gselfile.name,
      binary5 = 'scale-post'
      cmd5 = [binary5]
      cmd5 += [
        'ark:-',
        str(posterior_scale),
        'ark:|gzip -c >' + postfile.name,
      ]

      with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        pipe4 = Popen (cmd4, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe5 = Popen (cmd5, stdin=pipe4.stdout, stdout=PIPE, stderr=logfile)
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
          binary6 = 'ivector-extractor-acc-stats'
          cmd6 = [binary6]
          cmd6 += [
            '--num-threads=4',
            '--num-samples-for-weights=' + str(num_samples_for_weights),
            inModel,
            'ark:' + arkfile.name,
            'ark:gunzip -c ' + postfile.name + '|',
            accfile.name,
          ]
          # ark,s,cs

          with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
            pipe6 = Popen (cmd6, stdin=PIPE, stdout=PIPE, stderr=logfile)
            # io.write_mat(pipe6.stdin, feats, key='abc')
            pipe6.communicate()
            with open(logfile.name) as fp:
              logtxt = fp.read()
              logger.debug("%s", logtxt)
          
          binary7 = 'ivector-extractor-est'
          cmd7 = [binary7]
          with tempfile.NamedTemporaryFile(delete=False, suffix='.ie') as estfile:
            cmd7 += [
              '--num-threads=4',
              inModel,
              accfile.name,
              estfile.name,
            ]
            with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
              pipe7 = Popen (cmd7, stdin=PIPE, stdout=PIPE, stderr=logfile)
              pipe7.communicate()
              with open(logfile.name) as fp:
                logtxt = fp.read()
                logger.debug("%s", logtxt)

              os.unlink(inModel)
              inModel=estfile.name

  shutil.copyfile(inModel, projector_file+'.ie')
  os.unlink(inModel)

  return projector_file+'.ie'

def ivector_extract(feats, projector_file, num_gselect=20, min_post=0.025, posterior_scale=1.0):
  """ Implements egs/sre10/v1/extract_ivectors.sh
  """

  # import ipdb; ipdb.set_trace()
  # ivector-extract --verbose=2 $srcdir/final.ie "$feats" ark,s,cs:- \
  # ark,scp,t:$dir/ivector.JOB.ark,$dir/ivector.JOB.scp || exit 1;
  fgmm_model=projector_file+'.fubm'

  # Initialize the i-vector extractor using the FGMM input
  binary1 = 'fgmm-global-to-gmm'
  cmd1 = [binary1]
  with tempfile.NamedTemporaryFile(delete=False, suffix='.dubm') as dubmfile:
    cmd1 += [
      fgmm_model,
      dubmfile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe1.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

  binary = 'gmm-gselect'
  cmd = [binary]
  with tempfile.NamedTemporaryFile(suffix='.gsel') as gselfile:
    cmd += [
      '--n=' + str(num_gselect),
      dubmfile.name,
      'ark:-',
      'ark:' + gselfile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe = Popen (cmd, stdin=PIPE, stdout=PIPE, stderr=logfile)
      io.write_mat(pipe.stdin, feats, key='abc')
      pipe.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)
        
    with tempfile.NamedTemporaryFile(suffix='.post') as postfile:
      binary2 = 'fgmm-global-gselect-to-post'
      cmd2 = [binary2]
      cmd2 += [
        '--min-post=' + str(min_post),
        fgmm_model,
        'ark:-',
        'ark,s,cs:' + gselfile.name,
        'ark:-',
      ]
      binary3 = 'scale-post'
      cmd3 = [binary3]
      cmd3 += [
        'ark:-',
        str(posterior_scale),
        'ark:' + postfile.name,
      ]
      
      with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
        pipe2 = Popen (cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
        pipe3 = Popen (cmd3, stdin=pipe2.stdout, stdout=PIPE, stderr=logfile)
        io.write_mat(pipe2.stdin, feats, key='abc')
        pipe2.stdin.close()
        pipe3.communicate()
        with open(logfile.name) as fp:
          logtxt = fp.read()
          logger.debug("%s", logtxt)
          
      binary4 = 'ivector-extract'
      cmd4 = [binary4]
      cmd4 += [
        projector_file+'.ie',
        'ark:-',
        'ark,s,cs:' + postfile.name,
        'ark:-',
      ]
      # import ipdb; ipdb.set_trace()
      
      with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
        pipe4 = Popen (cmd4, stdin=PIPE, stdout=PIPE, stderr=logfile)
        io.write_mat(pipe4.stdin, feats, key='abc')
        pipe4.stdin.close()
        with open(logfile.name) as fp:
          logtxt = fp.read()
          logger.debug("%s", logtxt)

        # read ark from pipe1.stdout
        ret = [mat for name,mat in io.read_vec_flt_ark(pipe4.stdout)][0]
        return ret

def plda_train(feats, enroller_file):
  """ Implements egs/sre10/v1/plda_scoring.sh
  """
  # ivector-compute-plda ark:$plda_data_dir/spk2utt \
  # "ark:ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp  ark:- |" \
  # $plda_ivec_dir/plda || exit 1;

  logger.debug("-> PLDA calculation")
  # 1. Create Kaldi training data structure
  # import ipdb; ipdb.set_trace()
  with tempfile.NamedTemporaryFile(mode='w+t', suffix='.spk2utt', delete=False) as spkfile:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ark') as arkfile:
      i=0
      with open(arkfile.name,'wb') as f:
        for spk in feats:
          j=0
          spkid='spk'+str(i)
          spkfile.write(spkid)
          for utt in spk:
            # print i, j
            spkutt=spkid+'utt'+str(j)
            io.write_vec_flt(f, utt, key=spkutt)
            spkfile.write(' '+spkutt)
            j += 1
          spkfile.write("\n")
          i += 1

  binary1 = 'ivector-normalize-length'
  cmd1 = [binary1]
  cmd1 += [
    'ark:'+arkfile.name,
    'ark:-',
  ]
  binary2 = 'ivector-compute-plda'
  cmd2 = [binary2]
  
  with tempfile.NamedTemporaryFile(suffix='.plda') as pldafile:
    cmd2 += [
      'ark,t:'+spkfile.name,
      'ark:-',
      pldafile.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe2 = Popen (cmd2, stdin=pipe1.stdout, stdout=PIPE, stderr=logfile)
      pipe2.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

    shutil.copyfile(pldafile.name, enroller_file + '.plda')
        

  # compute global mean
  # ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
  # ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;
  # import ipdb; ipdb.set_trace()
  binary3 = 'ivector-mean'
  cmd3 = [binary3]
  with tempfile.NamedTemporaryFile(suffix='.mean') as meanfile:
    cmd3 += [
      'ark:-',
      meanfile.name,
      ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe3 = Popen (cmd3, stdin=pipe1.stdout, stdout=PIPE, stderr=logfile)
      pipe3.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

    shutil.copyfile(meanfile.name, enroller_file + '.plda.mean')

  os.unlink(spkfile.name)
  os.unlink(arkfile.name)

  return enroller_file + '.plda'

def plda_enroll(feats, enroller_file):
  """ Implements egs/sre10/v1/plda_scoring.sh
  """
  # ivector-normalize-length scp:$dir/ivector.scp  ark:- \| \
  # ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
  # ivector-normalize-length ark:- ark,scp:$dir/spk_ivector.ark,$dir/spk_ivector
  logger.debug("-> PLDA enrollment")
  # 1. Create Kaldi training data structure
  with tempfile.NamedTemporaryFile(mode='w+t', suffix='.spk2utt', delete=False) as spkfile:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ark') as arkfile:
      i=0
      with open(arkfile.name,'wb') as f:
        j=0
        spkid='spk'+str(i)
        spkfile.write(spkid)
        for utt in feats:
          # print i, j
          spkutt=spkid+'utt'+str(j)
          io.write_vec_flt(f, utt, key=spkutt)
          spkfile.write(' '+spkutt)
          j += 1
        spkfile.write("\n")
        i += 1

  binary1 = 'ivector-normalize-length'
  cmd1 = [binary1]
  cmd1 += [
    'ark:'+arkfile.name,
    'ark:-',
  ]
  binary2 = 'ivector-mean'
  cmd2 = [binary2]
  cmd2 += [
    'ark,t:'+spkfile.name,
    'ark:-',
    'ark:-',
  ]
  binary3 = 'ivector-normalize-length'
  cmd3 = [binary3]
  cmd3 += [
    'ark:-',
    'ark:-',
  ]
  binary4 = 'ivector-subtract-global-mean'
  cmd4 = [binary4]
  cmd4 += [
    enroller_file+'.plda.mean',
    'ark:-',
    'ark:-',
  ]
  binary5 = 'ivector-normalize-length'
  cmd5 = [binary5]
  with tempfile.NamedTemporaryFile(delete=False, suffix='.ark') as spkarkfile:
    cmd5 += [
      'ark:-',
      'ark:'+spkarkfile.name,
    ]
    # import ipdb; ipdb.set_trace()
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe2 = Popen (cmd2, stdin=pipe1.stdout, stdout=PIPE, stderr=logfile)
      pipe3 = Popen (cmd3, stdin=pipe2.stdout, stdout=PIPE, stderr=logfile)
      pipe4 = Popen (cmd4, stdin=pipe3.stdout, stdout=PIPE, stderr=logfile)
      pipe5 = Popen (cmd5, stdin=pipe4.stdout, stdout=PIPE, stderr=logfile)
      pipe5.communicate()
      with open(logfile.name, 'rt') as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

  os.unlink(spkfile.name)
  os.unlink(arkfile.name)

  return spkarkfile.name

def plda_score(feats, model, ubm):
  """ Implements egs/sre10/v1/plda_scoring.sh
  """
  # import ipdb; ipdb.set_trace()
  # ivector-plda-scoring --normalize-length=true \
  # --num-utts=ark:${enroll_ivec_dir}/num_utts.ark \
  # "ivector-copy-plda --smoothing=0.0 ${plda_ivec_dir}/plda - |" \
  # "ark:ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec scp:${enroll_ivec_dir}/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
  # "ark:ivector-normalize-length scp:${test_ivec_dir}/ivector.scp ark:- | ivector-subtract-global-mean ${plda_ivec_dir}/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  # "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/plda_scores || exit 1;

  logger.debug("-> PLDA scoring")
  # 1.

  binary1 = 'ivector-copy-plda'
  cmd1 = [binary1]

  # tests/probes
  binary2 = 'ivector-normalize-length'
  cmd2 = [binary2]
  cmd2 += [
    'ark:-',
    'ark:-',
  ]

  binary2 = 'ivector-subtract-global-mean'
  cmd2 = [binary2]
  cmd2 += [
    ubm+'.plda.mean',
    'ark:-',
    'ark:-',
  ]
  binary3 = 'ivector-normalize-length'
  cmd3 = [binary3]
  cmd3 += [
    'ark:-',
    'ark:-',
  ]

  # scoring
  binary4 = 'ivector-plda-scoring'
  cmd4 = [binary4]
  with tempfile.NamedTemporaryFile(mode='w+t', suffix='.trials', delete=False) as trials:
    trials.write("spk0 spk1\n")

  ret=0

  # plda smooting
  with tempfile.NamedTemporaryFile(delete=False, suffix='.plda') as plda:
    cmd1 += [
      '--smoothing=0.0',
      ubm+'.plda',
      plda.name,
    ]
    with tempfile.NamedTemporaryFile(suffix='.log') as logfile:
      pipe1 = Popen (cmd1, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe1.communicate()
      with open(logfile.name) as fp:
        logtxt = fp.read()
        logger.debug("%s", logtxt)

  with tempfile.NamedTemporaryFile(delete=False, suffix='.score') as score:
    cmd4 += [
      '--normalize-length=true',
      plda.name,
      'ark:'+model,
      'ark:-',
      trials.name,
      score.name,
    ]

    with tempfile.NamedTemporaryFile(suffix='.log') as logfile: 
      pipe2 = Popen (cmd2, stdin=PIPE, stdout=PIPE, stderr=logfile)
      pipe3 = Popen (cmd3, stdin=pipe2.stdout, stdout=PIPE, stderr=logfile)
      pipe4 = Popen (cmd4, stdin=pipe3.stdout, stdout=PIPE, stderr=logfile)
      io.write_vec_flt(pipe2.stdin, feats, key='spk1')
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
