datapath='/home/owner1/lre_jaga/test_data/lre09_30';
destpath='/home/owner1/lre_jaga/test_data/lre09_30_out';
dir1=dir(datapath);
folder=char(dir1.name);
folder=folder(3:end,:);
for i=1:size(folder,1)
    
    dir2=dir(strcat(datapath,'/',folder(i,:)));
    files=char(dir2.name);
    files=files(3:end,:);
    mkdir(strcat(destpath,'/',folder(i,:)));
    for j=1:size(files,1)
        
        
        audiopath=strcat(datapath,'/',folder(i,:),'/',files(j,:));
        [d,fs]=audioread(audiopath);
        
        if j<10
           audiodest=strcat(destpath,'/',folder(i,:),'/',folder(i,:),'-','000',num2str(j),'.wav'); 
        elseif j<100 && j>10
            audiodest=strcat(destpath,'/',folder(i,:),'/',folder(i,:),'-','00',num2str(j),'.wav');
        elseif j<1000 && j>100
            audiodest=strcat(destpath,'/',folder(i,:),'/',folder(i,:),'-','0',num2str(j),'.wav');
        elseif j>1000 
            audiodest=strcat(destpath,'/',folder(i,:),'/',folder(i,:),'-',num2str(j),'.wav');
        end
        audiowrite(audiodest,d,fs);    
    end
    
    
    
end