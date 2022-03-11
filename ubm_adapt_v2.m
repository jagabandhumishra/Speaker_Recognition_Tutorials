function[m v w]= ubm_adapt_v2(X,NG,Mean_UBM,Var_UBM,Weights_UBM)

% %     ubmpath='D:\MODEL_FEATURE_RESULT\SWITCH_BOARD\UBM_TRAIN\vop_vep\eps_200ms_constraint_0.05TH\128\';
% %     d=dir(ubmpath);
% %     files=char(d.name);
% %     
% %     for k= 3:length(d)
% %     path=strcat(ubmpath,files(k,:));
% %     load(path);
% %     end
% %     
  
    [T, d]=size(X);
    n=0;
    Exn=0;
    Ex2n=0;
   for t=1:T 
    xt=X(t,:)';
    
    for j=1:NG
        mi=Mean_UBM(j,:)';
        
        vi=eye(d);
        for k=1:d
          vi(k,k)=Var_UBM(j,k);
        end      
        
        p(j)= gauss_prob(xt,mi,vi);
    end
    
    wjpj=Weights_UBM'.*p;
    PrDr=sum(wjpj);
    
    Prt=wjpj/PrDr;
    
    n=n+Prt;
    Exn=Exn+ xt*Prt;
    Ex2n=Ex2n+(diag(xt*xt'))*Prt;
   end
   
   for j=1:NG
   Ex(:,j)=Exn(:,j)./n(j);
   Ex2(:,j)=Ex2n(:,j)./n(j);
   end
   
   alpha=n./(n+16);
   w_temp= (alpha.*n)/T+(1-alpha).*Weights_UBM';
   w=w_temp/sum(w_temp);
   
   for j=1:NG
       mu=Mean_UBM(j,:)';
       sigma2=Var_UBM(j,:)';
       
       
       mtemp=alpha(j)*Ex(:,j)+(1-alpha(j))*mu;
       m(:,j)=mtemp;
       
       
       v(:,j)=alpha(j)*Ex2(:,j)+(1-alpha(j))*(sigma2+diag(mu*mu'))- diag(mtemp*mtemp');
   end
   
   
   w=w';
   m=m';
   v=v';
