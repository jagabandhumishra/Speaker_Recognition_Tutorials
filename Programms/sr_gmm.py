# -*- coding: utf-8 -*-
"""SR_GMM.ipynb
Speaker Identification experiment

"""


import numpy as np
import librosa
#pip install librosa 
#%%
def extract_features(y,sr):

    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=14)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = mfcc[1:]
    mfcc_delta = mfcc_delta[1:]
    mfcc_delta2 = mfcc_delta2[1:]

    # print(mfcc.shape)
    # print(mfcc_delta.shape)
    # print(mfcc_delta2.shape)

    combined = np.hstack((mfcc.T,mfcc_delta.T, mfcc_delta2.T)) 
    return combined

#%%

import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import warnings
warnings.filterwarnings("ignore")


#path to training data
source   = "/home/iit/Speaker_Recognition_Tutorials/train_data/"   

#path where training speakers will be saved
dest = "speaker_models/"
#os.mkdir(dest)
arr = os.listdir(source)
print(arr)

sr=8000 # sampling frequency of the speech file
# Extracting features for each speaker 
features = np.asarray(())

for path in arr:    
    wavfile = os.listdir(source+str(path))
    print(path)
    print(wavfile)
    
    # read the audio
    y,sr= librosa.load(source + path + '/' + wavfile[0],sr=sr)
    # print(sr)

    features  = extract_features(y,sr)
    # print(features.shape)
    
    gmm = GMM(n_components = 16, max_iter=50, n_init = 3)
    gmm.fit(features)
        
    # dumping the trained gaussian model
    picklefile = path+".gmm"
    pickle.dump(gmm,open(dest + picklefile,'wb'))
    print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
    features = np.asarray(())

#%%

#path to testing data
source   = "/home/iit/Speaker_Recognition_Tutorials/test_data/"   

modelpath = "speaker_models/"

arr = os.listdir(source)
print(arr)
print()

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]
print(gmm_files)

#Load the Gaussian Models
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

print()
print(speakers)
print()

acc = 0
cnt = 0

# Read the test directory and get the list of test audio files 
for path in arr:   
    
  wavfiles = os.listdir(source+str(path))

  for  wavfile in wavfiles: 
    # read the audio
    print(wavfile)
    y,sr = librosa.load(source + path + '/' + wavfile,sr=sr)

    # extract 40 dimensional MFCC & delta MFCC features
    features  = extract_features(y,sr)
    # print(features.shape)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]         #checking with each model one by one
        scores = np.array(gmm.score(features))
        log_likelihood[i] = scores.sum()
    
    spk = np.argmax(log_likelihood)
    print("detected as - ", speakers[spk])
    print()

    cnt = cnt+1
    if(speakers[spk]==path):
      acc = acc+1

print('accuracy ', (acc/cnt)*100)

