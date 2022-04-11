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



nnet_dir=exp/xvector_nnet_1a

#local/split_long_utts.sh --max-utt-len 120 data/lredata data/lredata_train
utils/fix_data_dir.sh data/lredata
utils/validate_data_dir.sh --no-text --no-feats data/lredata




stage=1
# feature extraction for all training and test data
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset 
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  fi
  for name in lredata lre09_03 lre09_10 lre09_30; do
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

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/lredata/utt2num_frames > data/lredata/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
     Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python3 steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/lredata data/lredata_reverb
  cp data/lredata/vad.scp data/lredata_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/lredata_reverb data/lredata_reverb.new
  rm -rf data/lredata_reverb
  mv data/lredata_reverb.new data/lredata_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /home/owner1/lre_jaga/source_database/musan data   # change the path , give the folder path of MUSCAN

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python3 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/lredata data/lredata_noise
   #Augment with musan_music
  python3 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/lredata data/lredata_music
   #Augment with musan_speech
  python3 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/lredata data/lredata_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/lredata_aug data/lredata_reverb data/lredata_noise data/lredata_music data/lredata_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice    #doubt
  # the size of the SWBD+SRE list)

# 12800 also be changed to optimize the result, 12800 signifies no of augmented file you want to randomly add. 
  utils/subset_data_dir.sh data/lredata_aug 12800 data/lredata_aug_128k
  utils/fix_data_dir.sh data/lredata_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/lredata_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/lredata_combined data/lredata_aug_128k data/lredata

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/lredata_combined data/lredata_clan_combined   						#lredata_clan_combined=sre_combined
  utils/filter_scp.pl data/lredata/spk2utt data/lredata_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/lredata_clan_combined/utt2spk
  utils/fix_data_dir.sh data/lredata_clan_combined
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.

# prepare_feats_for_egs.sh and extract_xvectors.sh have to be changed for SDC use
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 5 --cmd "$train_cmd" \
    data/lredata_combined data/lredata_combined_no_sil exp/lredata_combined_no_sil
  utils/fix_data_dir.sh data/lredata_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/lredata_combined_no_sil/utt2num_frames data/lredata_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/lredata_combined_no_sil/utt2num_frames.bak > data/lredata_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/lredata_combined_no_sil/utt2num_frames data/lredata_combined_no_sil/utt2spk > data/lredata_combined_no_sil/utt2spk.new
  mv data/lredata_combined_no_sil/utt2spk.new data/lredata_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/lredata_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/lredata_combined_no_sil/spk2utt > data/lredata_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/lredata_combined_no_sil/spk2num | utils/filter_scp.pl - data/lredata_combined_no_sil/spk2utt > data/lredata_combined_no_sil/spk2utt.new
  mv data/lredata_combined_no_sil/spk2utt.new data/lredata_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/lredata_combined_no_sil/spk2utt > data/lredata_combined_no_sil/utt2spk

  utils/filter_scp.pl data/lredata_combined_no_sil/utt2spk data/lredata_combined_no_sil/utt2num_frames > data/lredata_combined_no_sil/utt2num_frames.new
  mv data/lredata_combined_no_sil/utt2num_frames.new data/lredata_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/lredata_combined_no_sil
fi

# for language recognition task lredata is used instead of sre_swbd data
# silence is removed for x vector training
# x vector is trained on mfcc feature 

# network structure can be changed in run_xvector.sh file
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 --data data/lredata_combined_no_sil --nnet-dir $nnet_dir --egs-dir $nnet_dir/egs			# change it commented by jagabandhu,   uncomment it for training x vector, the x vector will train for non silence speech frames.
 
if [ $stage -le 7 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.
  #sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 5 \
   # $nnet_dir data/sre16_major \
   # exp/xvectors_sre16_major

  # Extract xvectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.


# x-vector extracted for all data
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 5 \
    $nnet_dir data/lredata_clan_combined \
   exp/xvectors_lredata_clan_combined			#augmented data

sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 5 \
    $nnet_dir data/lredata \
   exp/xvectors_lredata					# clean data, used instead of sre16_major, sre16_enroll 

 sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 5 \
    $nnet_dir data/lre09_03 \
    exp/xvectors_lre09_03

 sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 5 \
    $nnet_dir data/lre09_10 \
    exp/xvectors_lre09_10


 sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 5 \
    $nnet_dir data/lre09_30 \
    exp/xvectors_lre09_30



  # The SRE16 test data
  #sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 5 \
    #$nnet_dir data/sre16_eval_test \
    #exp/xvectors_sre16_eval_test

  # The SRE16 enroll data
  #sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 5 \
  #  $nnet_dir data/sre16_eval_enroll \
   # exp/xvectors_sre16_eval_enroll
fi
if [ $stage -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_lredata/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_lredata/xvector.scp \
    exp/xvectors_lredata/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=10    							# should be less then the no of language class, can be reduced for optimizing result
  $train_cmd exp/xvectors_lredata_clan_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_lredata_clan_combined/xvector.scp ark:- |" \
    ark:data/lredata_clan_combined/utt2spk exp/xvectors_lredata_clan_combined/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd exp/xvectors_lredata_clan_combined/log/plda.log \
    ivector-compute-plda ark:data/lredata_clan_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_lredata_clan_combined/xvector.scp ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_lredata_clan_combined/plda || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  $train_cmd exp/xvectors_lredata/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/xvectors_lredata_clan_combined/plda \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_lredata/xvector.scp ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/xvectors_lredata/plda_adapt || exit 1;
fi

if [ $stage -le 9 ]; then
  # Get results using the out-of-domain PLDA model.
for dur in 03 10 30; do 

for name in aeng	cantonese_chinese	creol	farsi	georg	korea	mandn	rusia	spanish	turki	vietnamese; do

  $train_cmd exp/scores/log/lre09_${dur}_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_lredata/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_lredata_clan_combined/plda - |" \
    "ark:ivector-mean ark:data/lre09_${dur}/spk2utt scp:exp/xvectors_lre09_${dur}/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_lredata/mean.vec ark:- ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_lredata/mean.vec scp:exp/xvectors_lre09_${dur}/xvector.scp ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat 'data/lre09_${dur}/${name}_trail.txt' | cut -d\  --fields=1,2 |" exp/scores/${name}_lre09_${dur}_eval_scores || exit 1;

	pooled_eer=$(paste data/lre09_${dur}/${name}_trail.txt data/lre09_${dur}/${name}_grtr.txt exp/scores/${name}_lre09_${dur}_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)

	 echo "Using Out-of-Domain PLDA ${name} ${dur}, EER: Pooled ${pooled_eer}%"
done
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -			#print line
done
  
fi



if [ $stage -le 10 ]; then
  # Get results using the adapted PLDA model.
for dur in 03 10 30; do 

for name in aeng	cantonese_chinese	creol	farsi	georg	korea	mandn	rusia	spanish	turki	vietnamese; do

  $train_cmd exp/scores/log/lre09_${dur}_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_lredata/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_lredata/plda_adapt - |" \
    "ark:ivector-mean ark:data/lredata/spk2utt scp:exp/xvectors_lredata/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_lredata/mean.vec ark:- ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_lredata/mean.vec scp:exp/xvectors_lre09_${dur}/xvector.scp ark:- | transform-vec exp/xvectors_lredata_clan_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat 'data/lre09_${dur}/${name}_trail.txt' | cut -d\  --fields=1,2 |" exp/scores/${name}_lre09_${dur}_eval_scores_adapt || exit 1;


pooled_eer=$(paste data/lre09_${dur}/${name}_trail.txt data/lre09_${dur}/${name}_grtr.txt exp/scores/${name}_lre09_${dur}_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  
  echo "Using Adapted PLDA ${name} ${dur}, EER: Pooled ${pooled_eer}%"

done
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -   #print line
done

fi




