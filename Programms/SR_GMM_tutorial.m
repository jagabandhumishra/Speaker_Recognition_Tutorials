clc;
clear all;
close all;
%%  read files 
base_path='/home/jagabandhu/data/Speaker_Recognition_Tutorials';
train_path=strcat(base_path,'/','train_data');
test_path=strcat(base_path,'/','test_data');
%% read from wav files
dr=dir(train_path);
spks=char(dr.name);
spks=spks(3:end,:);
%%
cl_sz=64;

for i=1:length(spks)
    
    file_path=strcat(train_path,'/',spks(i,:));
    
    ddr=dir(file_path);
    fl_nm=char(ddr.name);
    fl_nm=fl_nm(3:end,:); % as one file has been taken for training
    
    wav_path=strcat(file_path,'/',fl_nm);
    [d,fs]=audioread(wav_path);
    d=d-mean(d);
    d=d./(1.01*(max(abs(d))));
    
    %% mfcc-feature extraction
    [MFCC,DMFCC,DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,14,24,20,10,1,1,2);
    
    mfcc=[MFCC(:,2:end), DMFCC(:,2:end),DDMFCC(:,2:end)]; %% Discard c0 co-effitient
   %% Training
    [gmm.m,gmm.v,gmm.w]=gmmtrain_EMalgor_V1(mfcc,50,cl_sz); %% GMM modelling
    
    GMMSPK{i}=gmm;
    disp(['training for SPK ',num2str(spks(i,:)),' completed'])
    
end

%% Testing

test_path=strcat(base_path,'/','test_data');
dr=dir(test_path);
tspks=char(dr.name);
tspks=tspks(3:end,:);

%% testing spk wise
ll=1;
true_scores=[];
false_scores=[];

for i=1:length(tspks)
    
    file_path=strcat(test_path,'/',tspks(i,:));
    
    ddr=dir(file_path);
    fl_nm=char(ddr.name);
    fl_nm=fl_nm(3:end,:); % as one file has been taken for training
    
    
    
    for j=1:size(fl_nm,1)
      
        %% read test wav files
     spk_idx(ll)=i;    
     wav_path=strcat(file_path,'/',fl_nm(j,:)); 
     [d,fs]=audioread(wav_path);
     d=d-mean(d);
     d=d./(1.01*(max(abs(d))));
   
    %% mfcc-feature extraction
    [MFCC,DMFCC,DDMFCC]=mfcc_delta_deltadelta_rasta_v5(d,fs,14,24,20,10,1,1,2);
    
    mfcc=[MFCC(:,2:end), DMFCC(:,2:end),DDMFCC(:,2:end)]; %% Discard c0 co-effitient
    
    
    for k=1:length(GMMSPK)
    
    gmm_spk=GMMSPK{k};
        
    llk(ll,k)=mean(gmmlpdf_voicebox_pkm_v1(mfcc,gmm_spk.m,gmm_spk.v,gmm_spk.w));
    
    if i==k
        true_scores=[true_scores;llk(ll,k)];
    else
       false_scores=[false_scores;llk(ll,k)]; 
    
    end
    
    end
     ll=ll+1;
     disp(['testing for SPK ',num2str(tspks(i,:)),' having utt ',fl_nm(j,:),' is completed'])
    end
    
end
%% identification accuracy

[~,pred_spk]=max(llk,[],2);
acc=(length(find(pred_spk==spk_idx'))/(length(spk_idx))*100)

%% FA and FR computation (verification)
scores=[true_scores;false_scores];
th=0.982*mean(scores);

FR=length(find(true_scores<=th))
FA=length(find(false_scores>th))

FRR=(FR/length(true_scores))*100
FAR=(FA/length(false_scores))*100

VER=0.5*(FRR+FAR) % verification error rate
%% EER (Equal Error rate)
EER=plot_EER(true_scores,false_scores,'.')