function distance=finddistance(centroid,lprcoeff);

[no_frames,N_dim]=size(lprcoeff);

centroid=centroid';

[len_codewords,l]=size(centroid);

for m=1:no_frames

    for c=1:len_codewords
    
        framedist(1,c)=sqrt(sum((lprcoeff(m,:)-centroid(c,:)).^2));
    
    end

    filedist(1,m)=min(framedist);

end

distance=mean(filedist);