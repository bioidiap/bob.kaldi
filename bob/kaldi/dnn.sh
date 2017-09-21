#!/usr/bin/zsh
#

# # Allow setshell
# software=/idiap/resource/software
# source $software/initfiles/shrc $software
# SETSHELL kaldi

out=test/kaldi
if [[ ! -e $out ]]; then
    mkdir $out
fi

nnet-forward --feature-transform=dnn/ami.feature_transform \
	     dnn/ami.nnet \
	     ark,t:$out/sample16k.cmvn.deltas.ark \
	     ark,t:$out/sample16k.posteriors.ark

