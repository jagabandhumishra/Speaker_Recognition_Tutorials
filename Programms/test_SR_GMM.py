#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:29:16 2022

@author: iit
"""

import pickle
import os
import numpy as np
import librosa as lb
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import warnings
warnings.filterwarnings("ignore")


#%% 
def extract_features(y,sr):

    mfcc = lb.feature.mfcc(y=y, sr=sr,n_mfcc=14)
    mfcc_delta = lb.feature.delta(mfcc)
    mfcc_delta2 = lb.feature.delta(mfcc, order=2)

    mfcc = mfcc[1:]
    mfcc_delta = mfcc_delta[1:]
    mfcc_delta2 = mfcc_delta2[1:]

    # print(mfcc.shape)
    # print(mfcc_delta.shape)
    # print(mfcc_delta2.shape)

    combined = np.hstack((mfcc.T,mfcc_delta.T, mfcc_delta2.T)) 
    return combined

#%% 
def verify(score,th):
    if score>=th:
        op=1
    else:
        op=0
    return op
#%%

path2str='/home/iit/Speaker_Recognition_Tutorials/Demo_sys/test_data/'
id='JMM28'
str_data_spk=path2str+id
os.mkdir(str_data_spk)
sr=8000
#%%
wavfilepath=str_data_spk+'/tst_'+id+'.wav'
y,sr=lb.load(wavfilepath,sr=sr)
features  = extract_features(y,sr)

model_load="/home/iit/Speaker_Recognition_Tutorials/Demo_sys/train_data/" 

model_load_path=model_load+id+'/'+id+'.gmm'
model=pickle.load(open(model_load_path,'rb'))

score=model.score(features)


#%% verify
th=-100
op=verify(score,th)
print(op)




#%%