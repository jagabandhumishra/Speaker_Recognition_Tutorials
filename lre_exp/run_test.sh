#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.
#Edited by jagabandhu mishra for language recognition using x-vector 

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# lre trials

nnet_dir=exp/xvector_nnet_1a

#local/split_long_utts.sh --max-utt-len 120 data/lredata data/lredata_train
utils/fix_data_dir.sh data/lredata
utils/validate_data_dir.sh --no-text --no-feats data/lredata




stage=1

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  fi
  for name in lre09_03 lre09_10 lre09_30 ; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  #utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre
  #utils/fix_data_dir.sh data/lredata
fi
