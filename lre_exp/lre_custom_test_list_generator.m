clc;
clear all;
close all;
basedir='/home/owner1/kaldi/egs/lre_exp/data/lre09_30';
file=strcat(basedir,'/utt2lang');

fileID = fopen(file);
protocol = textscan(fileID, '%s%s');
fclose(fileID);

uttId=protocol{1};
model=protocol{2};
umodel=unique(model);
domy='a';


for i=1:size(umodel,1)
    
flname=strcat(basedir,'/',umodel{i},'_trail.txt'); 
 flname1=strcat(basedir,'/',umodel{i},'_grtr.txt');   
fid = fopen(flname,'wt');
fid1= fopen (flname1,'wt');

for j=1:size(uttId,1)
    
 mod= umodel{i};  
ind=find(uttId{j}=='-');
seg=uttId{j};
seg=seg(1:ind(1)-1);
fprintf(fid, '%s\t%s\n',umodel{i},uttId{j});

if strcmp(mod,seg)
    grtr='target';
%fprintf(fid, '%s\t%s\t%s\t%s\n',umodel{i},uttId{j},domy,grtr);
fprintf(fid1, '%s\n',grtr);
else
%fprintf(fid, '%s\t%s\t%s\t%s\n',umodel{i},uttId{j},domy,'nontarget');
fprintf(fid1, '%s\n','nontarget');
end


end
fclose(fid);
fclose(fid1);
end