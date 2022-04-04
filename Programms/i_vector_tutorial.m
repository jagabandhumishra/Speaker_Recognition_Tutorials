clc;
clear all;
close all;
%% load ubm model
ubm_model_path='/home/jagabandhu/data/Speaker_Recognition_Tutorials/model_save/ubm_64.mat';
load(ubm_model_path);

base_path='/home/jagabandhu/data/Speaker_Recognition_Tutorials';
train_path=strcat(base_path,'/','train_data');
test_path=strcat(base_path,'/','test_data');
ubm_path=strcat(base_path,'/','ubm_male_data');
dest_path=strcat(base_path,'/','model_save');
%% supervector creation
m=[];E=[];w=[];
[ro, co]=size(ubm.m);
n_mixtures=ro;
dim=co;
for i=1:ro
    m=[m;ubm.m(i,:)'];
    E=[E;ubm.v(i,:)'];
    w=[w;ubm.w(i,:)'];
end
%% suffitient statistics estimation
% for ubm/development data
dr=dir(ubm_path);
spks=char(dr.name);
spks=spks(3:end,:);
disp('Suffitent statistic estimation for ubm/dev data')
N = zeros(size(spks,1), n_mixtures);
F = zeros(size(spks,1), n_mixtures * dim);
%%
for i=1:length(spks)
    
    file_path=strcat(ubm_path,'/',spks(i,:));
    
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
    
    [Ni, Fi] = collect_suf_stats(mfcc', ubm.m', ubm.v', w);
   N(i,:) = Ni;
   F(i,:) = Fi;
    
end

%% T-matrix training
m1=m';E1=E';w1=w';

nw=200;
itr_T=10;

disp(['Initializing T matrix (randomly)'])
T = randn(nw, size(F, 2)) * sum(E1,2) * 0.001;

% we don't use the second order stats - set 'em to empty matrix
S = [];

spk_ids = (1:size(N,1))';

n_speakers=max(spk_ids);
n_sessions=size(spk_ids,1);

for ii=1:itr_T
    disp(' ')
    disp(['Starting iteration: ' num2str(ii)])
    
    ct=cputime;
    et=clock;
    
    [~,T]=estimate_y_and_v(F, N, S, m1, E1, 0, T, 0, zeros(n_speakers,1), 0, zeros(n_sessions,1), spk_ids);
    
    disp(['Iteration: ' num2str(ii) ' CPU time:' num2str(cputime-ct) ' Elapsed time: ' num2str(etime(clock,et))])

end

%% extract i-vectors from training set utterances.

% suffitient statistics estimation
% for ubm/development data
dr=dir(train_path);
spks=char(dr.name);
spks=spks(3:end,:);
disp('Suffitent statistic estimation for train data')
trn.N = zeros(size(spks,1), n_mixtures);
trn.F = zeros(size(spks,1), n_mixtures * dim);

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
    
    [Ni, Fi] = collect_suf_stats(mfcc', ubm.m', ubm.v', w);
   trn.N(i,:) = Ni;
   trn.F(i,:) = Fi;
    
end
%% training i-vector extraction

trn.spk_ids       = (1:size(trn.N,1))';
n_speakers=max(trn.spk_ids);
n_sessions=size(trn.spk_ids,1);

% we don't use the D matrix
d = zeros(1, size(trn.F, 2));


% initialize the factors
wt      = zeros(n_speakers,nw);
trn.z   = zeros(n_speakers,size(trn.F,2));
junk    = zeros(n_sessions,1);


wt = estimate_y_and_v(trn.F, trn.N, 0, m1, E1, d, T, 0, trn.z, w, junk, trn.spk_ids);

%% test i-vectors

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
    [Ni, Fi] = collect_suf_stats(mfcc', ubm.m', ubm.v', w);
    tst.N=Ni';
    
    tst.F=Fi';
    
    % we don't use the D matrix
    d = zeros(1, size(tst.F, 2));


    % initialize the factors
    wtst      = zeros(1,nw);
    tst.z   = zeros(1,size(tst.F,2));
    junk    = zeros(1,1);


    wtst = estimate_y_and_v(tst.F, tst.N, 0, m1, E1, d, T, 0, trn.z, w, junk, [1]);

    
    
    
    
    
    
    for k=1:size(wt,1)
    
  
    
    
    cdis(ll,k)=(wt(k,:)*wtst')/(norm(wt(k,:))*norm(wtst));
    
    if i==k
        true_scores=[true_scores;cdis(ll,k)];
    else
       false_scores=[false_scores;cdis(ll,k)]; 
    
    end
    
    end
     ll=ll+1;
     disp(['testing for SPK ',num2str(tspks(i,:)),' having utt ',fl_nm(j,:),' is completed'])
    end
    
end
%%
[~,pred_spk]=max(cdis,[],2);
acc=(length(find(pred_spk==spk_idx'))/(length(spk_idx))*100)

%% LDA 
n_top_dirct=4;
dvlp_w=wt';

a=cellstr(spks);
b=1:5;
seq_id=1:5;

%==========================================================================
% Within class covariance
%==========================================================================
[n,P]=size(dvlp_w);
c=length(unique(seq_id));
cls_strt_indx=1;

S_w=0;
data_mean=mean(dvlp_w,2);
class_means=zeros(n,c);
class_sizes=zeros(1,c);
for i=1:c
    cls_inds=find(seq_id==i);
    class_data=dvlp_w(:,cls_inds);
    class_sizes(i)=size(class_data,2);
    S_w=S_w+cov(class_data',1)*class_sizes(i); 
    class_means(:,i)=mean(class_data,2);
end

%==========================================================================
% Between class covariance
%==========================================================================
S_b=0;
for i=1:c
    class_mean=class_means(:,i);
    S_b=S_b+((class_mean-data_mean)*(class_mean-data_mean)'*class_sizes(i));
end

inv_S_w_S_b=inv(S_w)*S_b;
[V, D]=eig(inv_S_w_S_b);



V_top=V(:,1:n_top_dirct);
lda_A=V_top;
%% WCCN
[n,P]=size(dvlp_w);
c=length(unique(seq_id));
cls_strt_indx=1;
dvlp_w=lda_A'*dvlp_w;
W=0;        
class_means=zeros(n,c);
class_sizes=zeros(1,c);
for i=1:c
    cls_inds=find(seq_id==i);
    class_data=dvlp_w(:,cls_inds);
    class_sizes(i)=size(class_data,2);
    W=W+cov(class_data',1)*class_sizes(i);        
end
W=W/c;

wccn_B=chol(inv(W),'lower');

%% testing spk wise
ll=1;
true_scores=[];
false_scores=[];
wt=wt*lda_A;
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
    [Ni, Fi] = collect_suf_stats(mfcc', ubm.m', ubm.v', w);
    tst.N=Ni';
    
    tst.F=Fi';
    
    % we don't use the D matrix
    d = zeros(1, size(tst.F, 2));


    % initialize the factors
    wtst      = zeros(1,nw);
    tst.z   = zeros(1,size(tst.F,2));
    junk    = zeros(1,1);


    wtst = estimate_y_and_v(tst.F, tst.N, 0, m1, E1, d, T, 0, trn.z, w, junk, [1]);

    
    wtst=wtst*lda_A;
    
    
    
    
    
    for k=1:size(wt,1)
    
  
    
    
    cdis(ll,k)=(wt(k,:)*wtst')/(norm(wt(k,:))*norm(wtst));
    
    if i==k
        true_scores=[true_scores;cdis(ll,k)];
    else
       false_scores=[false_scores;cdis(ll,k)]; 
    
    end
    
    end
     ll=ll+1;
     disp(['testing for SPK ',num2str(tspks(i,:)),' having utt ',fl_nm(j,:),' is completed'])
    end
    
end
%%
[~,pred_spk]=max(cdis,[],2);
acc=(length(find(pred_spk==spk_idx'))/(length(spk_idx))*100)


