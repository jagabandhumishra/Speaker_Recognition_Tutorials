function [m,v,w,g,f,pp,gg]=gmmtrain_EMalgor_V1(X,L,NG)

% Some of the Programs are taken from Voicebox
%--------------------------------------------------------------------------
% Inputs

% X   -> No of Feaature vector X dimension of each feature vector
% L   -> No of Iterations  (Recomended to use >10)
% NG  -> No of Gaussians

% Outputs: 

%     M   Mixture means, one row per mixture. 
%     V   Mixture variances, one row per mixture. 
%     W   Mixture weights, one per mixture. The weights will sum to unity. 
%     G   Average log probability of the input data points.
%     F   Fisher's Discriminant measures how well the data divides into classes.
%         It is the ratio of the between-mixture variance to the average mixture variance: a high value means the classes (mixtures) are well separated.
%     PP  Log probability of each data point
%     GG  Average log probabilities at the beginning of each iteration and at the end


% PKM on 09/06/2008 & 10/06/2008 & 16/06/2008
%--------------------------------------------------------------------------

[Mu,Sigma,Priors]=init_kmeans_v1(X,NG);

[m,v,w,g,f,pp,gg]=gmm_voicebox_pkm_V1(X,[],L,Mu,Sigma,Priors);


%***************************** ADDED BY PKM ON 09/06/2008 TO AVIOD RANDOM INITIALIZATION ******************

function [Mu,Sigma,Prioriw]=init_kmeans_v1(x,No_of_Gaussians)


       [centriod,Indexs,Prioriw,DistHist]=vqsplit(x',No_of_Gaussians);

        for i=1:No_of_Gaussians
            
 
            Indexs2=find(Indexs==i);
            
            if(length(Indexs2)==1)

            Data1=x(Indexs2,:);
            Mu(i,:)=(Data1);                       
            Sigma(i,:)=(Data1);                     
                
            else
                
            Data1=x(Indexs2,:);
            Mu(i,:)=mean(Data1);                       
            Sigma(i,:)=var(Data1);                     
            
            end
            
        end
        
        Prioriw=Prioriw';
%---------------------------------------------------------------------------------------------------------        
        
        
function [m,v,w,g,f,pp,gg]=gmm_voicebox_pkm_V1(x,c,l,m0,v0,w0)

%disp('******************** GMM MODELLING (VOICE BOX) ********************')

%GAUSSMIX fits a gaussian mixture pdf to a set of data observations [m,v,w,g,f]=(x,xv,l,m0,v0,w0)
%
% Inputs: n data values, k mixtures, p parameters, l loops
%
%     X(n,p)   Input data vectors, one per row.
%     c(1,p)   Minimum variance (can be a scalar if all components are identical or 0 if no minimum).
%              Use [] to take default value of var(x)/n^2
%     L        The integer portion of l gives a maximum loop count. The fractional portion gives
%              an optional stopping threshold. Iteration will cease if the increase in
%              log likelihood density per data point is less than this value. Thus l=10.001 will
%              stop after 10 iterations or when the increase in log likelihood falls below
%              0.001.
%              As a special case, if L=0, then the first three outputs are omitted.
%              Use [] to take default value of 100.0001
%     M0(k,p)  Initial mixture means, one row per mixture.
%     V0(k,p)  Initial mixture variances, one row per mixture.
%     W0(k,1)  Initial mixture weights, one per mixture. The weights should sum to unity.
%
%     Alternatively, if initial values for M0, V0 and W0 are not given explicitly:
%
%     M0       Number of mixtures required
%     V0       Initialization mode:
%                'f'    Initialize with K randomly selected data points [default]
%                'p'    Initialize with centroids and variances of random partitions
%                'k'    k-means algorithm ('kf' and 'kp' determine initialization of kmeans)
%                'h'    k-harmonic means algorithm ('hf' and 'hp' determine initialization of kmeans)
%              Mode 'hf' generally gives the best results but 'f' [the default] is faster
%
% Outputs: (Note that M, V and W are omitted if L==0)
%
%     M(k,p)   Mixture means, one row per mixture. (omitted if L==0)
%     V(k,p)   Mixture variances, one row per mixture. (omitted if L==0)
%     W(k,1)   Mixture weights, one per mixture. The weights will sum to unity. (omitted if L==0)
%     G       Average log probability of the input data points.
%     F        Fisher's Discriminant measures how well the data divides into classes.
%              It is the ratio of the between-mixture variance to the average mixture variance: a
%              high value means the classes (mixtures) are well separated.
%     PP(n,1)  Log probability of each data point
%     GG(l+1,1) Average log probabilities at the beginning of each iteration and at the end

%  Bugs/Suggestions
%     (2) Allow processing in chunks by outputting/reinputting an array of sufficient statistics
%     (3) Implement full covariance matrices
%     (5) Should scale before finding initial centres
%     (6) Other initialization options:
%              's'    scale dimensions to equal variance when initializing
%              'l'    LBG algorithm
%              'm'    Move-means (dog-rabbit) algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n,p]=size(x);
x2=x.^2;            % need x^2 for variance calculation
if ~length(c)
    c=var(x,1)/n^2;
end
if ~length(l)
    l=100+1e-4;
end
if nargin<6             % no initial values specified for m0, v0, w0
    k=m0;
    if n<=k             % each data point can have its own mixture
        m=x(mod((1:k)-1,n)+1,:);    % just include all points several times
        v=zeros(k,p);               % will be set to floor later
        w=zeros(k,1);
        w(1:n)=1/n;
        if l>0
            l=0.1;            % no point in iterating
        end
    else
        if nargin<5
            v0='f';         % default initialization mode
        end
        m=zeros(k,p);
        v=ones(k,p);
        w=repmat(1/k,k,1);
        if any(v0=='k')                     % k-means initialization
            if any(v0=='p')
                [m,e,j]=kmeans(x,k,'p');
            else
                [m,e,j]=kmeans(x,k,'f');
            end
            for i=1:k
                v(i,:)=var(x(j==i,:),1);
            end    
        elseif any(v0=='h')                     % k-harmonic means initialization
            if any(v0=='p')
                [m,e,j]=kmeanhar(x,k,'p');
            else
                [m,e,j]=kmeanhar(x,k,'f');
            end
            for i=1:k
                v(i,:)=var(x(j==i,:),1);
            end    
            
        elseif any(v0=='p')                  % Initialize using a random partition
            ix=ceil(rand(1,n)*k);       % allocate to random clusters
            ix(rnsubset(k,n))=1:k;      % but force at least one point per cluster
            for i=1:k
                m(i,:)=mean(x(ix==i,:),1);
            end
        else                                % Forgy initialization: choose k random points [default] 
            m=x(rnsubset(k,n),:);         % sample k centres without replacement
        end
    end
else
    k=size(m0,1);
    m=m0;
    v=v0;
    w=w0;
end
if length(c)>1          % if c is a row vector, turn it into a full matrix so it works with max()
    c=c(ones(k,1),:);
end
% size(v)
% size(c)

v=max(v,c);         % apply the lower bound



% If data size is large then do calculations in chunks

memsize=voicebox('memsize'); 
nb=min(n,max(1,floor(memsize/(8*p*k))));    % chunk size for testing data points
nl=ceil(n/nb);                  % number of chunks
jx0=n-(nl-1)*nb;                % size of first chunk

im=repmat(1:k,1,nb); im=im(:);
th=(l-floor(l))*n;
sd=(nargout > 3*(l~=0)); % = 1 if we are outputting log likelihood values
l=floor(l)+sd;   % extra loop needed to calculate final G value

lpx=zeros(1,n);             % log probability of each data point
wk=ones(k,1);
wp=ones(1,p);
wnb=ones(1,nb);
wnj=ones(1,jx0);

% EM loop

g=0;                           % dummy initial value for comparison
gg=zeros(l+1,1);
ss=sd;                       % initialize stopping count (0 or 1)
for j=1:l
    g1=g;                    % save previous log likelihood (2*pi factor omitted)
    m1=m;                       % save previous means, variances and weights
    v1=v;
    w1=w;
    vi=v.^(-1);                 % calculate quantities that depend on the variances
    vm=sqrt(prod(vi,2)).*w;
    vi=-0.5*vi;
    
    % first do partial chunk
    
    jx=jx0;
    ii=1:jx;
    kk=repmat(ii,k,1);
    km=repmat(1:k,1,jx);
    py=reshape(sum((x(kk(:),:)-m(km(:),:)).^2.*vi(km(:),:),2),k,jx);
    mx=max(py,[],1);                % find normalizing factor for each data point to prevent underflow when using exp()
    px=exp(py-mx(wk,:)).*vm(:,wnj);  % find normalized probability of each mixture for each datapoint
    ps=sum(px,1);                   % total normalized likelihood of each data point
    px=px./ps(wk,:);                % relative mixture probabilities for each data point (columns sum to 1)
    lpx(ii)=log(ps)+mx;
    pk=sum(px,2);                   % effective number of data points for each mixture (could be zero due to underflow)
    sx=px*x(ii,:);
    sx2=px*x2(ii,:);
    ix=jx+1;
    
    for il=2:nl
        jx=jx+nb;        % increment upper limit
        ii=ix:jx;
        kk=repmat(ii,k,1);
        py=reshape(sum((x(kk(:),:)-m(im,:)).^2.*vi(im,:),2),k,nb);
        mx=max(py,[],1);                % find normalizing factor for each data point to prevent underflow when using exp()
        px=exp(py-mx(wk,:)).*vm(:,wnb);  % find normalized probability of each mixture for each datapoint
        ps=sum(px,1);                   % total normalized likelihood of each data point
        px=px./ps(wk,:);                % relative mixture probabilities for each data point (columns sum to 1)
        lpx(ii)=log(ps)+mx;
        pk=pk+sum(px,2);                   % effective number of data points for each mixture (could be zero due to underflow)
        sx=sx+px*x(ii,:);
        sx2=sx2+px*x2(ii,:);
        ix=jx+1;
    end
    g=sum(lpx);                    % total log probability summed over all data points
    gg(j)=g;
    w=pk/n;                         % normalize to get the weights
    Tii=isnan(pk);
    pk(find(Tii==1))=0;
    if pk                           % if all elements of pk are non-zero
        m=sx./pk(:,wp);
        v=sx2./pk(:,wp);
    else
        wm=pk==0;                       % mask indicating mixtures with zero weights
        [vv,mk]=sort(lpx);             % find the lowest probability data points
        m=zeros(k,p);                   % initialize means and variances to zero (variances are floored later)
        v=m;
        m(wm,:)=x(mk(1:sum(wm)),:);                % set zero-weight mixture means to worst-fitted data points
        wm=~wm;                         % mask for non-zero weights
        m(wm,:)=sx(wm,:)./pk(wm,wp);  % recalculate means and variances for mixtures with a non-zero weight
        v(wm,:)=sx2(wm,:)./pk(wm,wp);
    end
    v=max(v-m.^2,c);                % apply floor to variances
    
    if g-g1<=th && j>1
        if ~ss break; end  %  stop 
        ss=ss-1;       % stop next time
    end
    
end
if sd  % we need to calculate the final probabilities
    pp=lpx'-0.5*p*log(2*pi);   % log of total probability of each data point
    gg=gg(1:j)/n-0.5*p*log(2*pi);    % average log prob at each iteration
    g=gg(end);
    %     gg' % *** DEBUG ***
    m=m1;       % back up to previous iteration
    v=v1;
    w=w1;
    mm=sum(m,1)/k;
    f=prod(sum(m.^2,1)/k-mm.^2)/prod(sum(v,1)/k);
end
if l==0         % suppress the first three output arguments if l==0
    m=g;
    v=f;
    w=pp;
end    




function [x,g,gg] = kmeanhar(d,k,l,e,x0)
%KMEANS Vector quantisation using K-harmonic means algorithm [X,ESQ,J]=(D,K,X0)
%
%  Inputs:
%
%    D(N,P)  contains N data vectors of dimension P
%    K       is number of centres required
%    L       integer portion is max loop count, fractional portion
%            gives stopping threshold as fractional reduction in performance criterion
%    E       is exponent in the cost function. Significantly faster if this is an even integer. [default 4]
%    X0(K,P) are the initial centres (optional)
%     
%      or alternatively
%
%    X0      gives the initialization method
%            'f'   pick K random elements of D as the initial centres [default]
%            'p'   randomly divide D into K sets and choose the centroids
%
%  Outputs:
%
%    X(K,P)  is output row vectors
%    G       is the final performance criterion value (normalized by N)
%    GG(L+1) value of performance criterion before each iteration and at end
%
% It is often a good idea to scale the input data so that it has equal variance in each
% dimension before calling KMEANHAR.

%  [1] Bin Zhang, "Generalized K-Harmonic Means - Boosting in Unsupervised Learning",
%      Hewlett-Packartd Labs, Technical Report HPL-2000-137, 2000 [Zhang2000]
%      http://www.hpl.hp.com/techreports/2000/HPL-2000-137.pdf

%  Bugs:
%      (1) Could use nested blocking to allow very large data arrays
%      (2) Could then allow incremental calling with partial data arrays (but messy)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sort out the input arguments

if nargin<5
    x0='f';
    if nargin<4
        e=[];
        if nargin<3
            l=[]
        end
    end
end
if ~length(e)
    e=4;  % default value
end
if ~length(l)
    l=50+1e-3; % default value
end
sd=5;       % number of times we must be below threshold


% split into chunks if there are lots of data points

memsize=voicebox('memsize'); 
[n,p] = size(d);
nb=min(n,max(1,floor(memsize/(8*p*k))));    % block size for testing data points
nl=ceil(n/nb);                  % number of blocks

% initialize if X0 argument is not supplied

if ischar(x0)
    if k<n
        if any(x0=='p')                  % Initialize using a random partition
            ix=ceil(rand(1,n)*k);       % allocate to random clusters
            ix(rnsubset(k,n))=1:k;      % but force at least one point per cluster
            x=zeros(k,p);
            for i=1:k
                x(i,:)=mean(d(ix==i,:),1);
            end
        else                                % Forgy initialization: choose k random points [default] 
            x=d(rnsubset(k,n),:);         % sample k centres without replacement
        end
    else
        x=d(mod((1:k)-1,n)+1,:);    % just include all points several times
    end
else
    x=x0;
end
eh=e/2;
th=l-floor(l);
l=floor(l)+(nargout>1);   % extra loop needed to calculate final performance value
if l<=0
    l=100;      % max number of iterations ever
end
if th==0
    th=-1;      % prevent any stopping if l has no fractional part
end
gg=zeros(l+1,1);
im=repmat(1:k,1,nb); im=im(:);

% index arrays for replication

wk=ones(k,1);
wp=ones(1,p);
wn=ones(1,n);
% 
% % Main calculation loop
% 
% We have the following relationships to [1] where i and k index
% the data values and cluster centres respectively:
%
%   This program     [Zhang2000] 
%
%     d(i,:)            x_i                     input data
%     x(k,:)            m_k                     cluster centres
%     py(k,i)           (d_ik)^2
%     dm(i)'            d_i,min^2
%     pr(k,i)           (d_i,min/d_ik)^2
%     pe(k,i)           (d_i,min/d_ik)^p
%     qik(k,i)          q_ik
%     qk(k)             q_k
%     qik(k,i)./qk(k)   p_ik 
%     se(i)'            d_i,min^p * sumk(d_ik^-p)
%     xf(i)'            d_i,min^-2 / sumk(d_ik^-p)
%     xg(i)'            d_i,min^-(p+2) / sumk(d_ik^-p)^2


ss=sd+1;        % one extra loop at the start
g=0;                % dummy initial value of g
for j=1:l
    
    g1=g;                           % save old performance
    x1=x;                           % save old centres
    % first do partial chunk
    
    jx=n-(nl-1)*nb;
    ii=1:jx;
    kx=repmat(ii,k,1);
    km=repmat(1:k,1,jx);
    py=reshape(sum((d(kx(:),:)-x(km(:),:)).^2,2),k,jx);
    dm=min(py,[],1);                 % min value in each column gives nearest centre
    dmk=dm(wk,:);                   % expand into a matrix
    dq=py>dmk;                      % update only these values
    pr=ones(k,jx);                   % leaving others at 1
    pr(dq)=dmk(dq)./py(dq);            % ratio of min(py)./py
    pe=pr.^eh;
    se=sum(pe,1);
    xf=dm.^(eh-1)./se;
    g=xf*dm.';                     % performance criterion (divided by k)
    xg=xf./se;
    qik=xg(wk,:).*pe.*pr;           % qik(k,i) is equal to q_ik in [Zhang2000]
    qk=sum(qik,2);      
    xs=qik*d(ii,:);
    ix=jx+1;
    for il=2:nl
        jx=jx+nb;        % increment upper limit
        ii=ix:jx;
        kx=ii(wk,:);
        py=reshape(sum((d(kx(:),:)-x(im,:)).^2,2),k,nb);
        dm=min(py,[],1);                 % min value in each column gives nearest centre
        dmk=dm(wk,:);                   % expand into a matrix
        dq=py>dmk;                      % update only these values
        pr=ones(k,nb);                   % leaving others at 1
        pr(dq)=dmk(dq)./py(dq);            % ratio of min(py)./py
        pe=pr.^eh;
        se=sum(pe,1);
        xf=dm.^(eh-1)./se;
        g=g+xf*dm.';                     % performance criterion (divided by k)
        xg=xf./se;
        qik=xg(wk,:).*pe.*pr;           % qik(k,i) is equal to q_ik in [Zhang2000]
        qk=qk+sum(qik,2);      
        xs=xs+qik*d(ii,:);
        ix=jx+1;
    end  
    gg(j)=g;
    x=xs./qk(:,wp);
    if g1-g<=th*g1
        ss=ss-1;
        if ~ss break; end  %  stop if improvement < threshold for sd consecutive iterations
    else
        ss=sd;
    end
end
gg=gg(1:j)*k/n;                       % scale and trim the performance criterion vector
g=g(end);
% gg' % *** DEBUIG ***
if nargout>1
    x=x1;                               % go back to the previous x values if G value is output
end




function [x,g,j,gg] = kmeans(d,k,x0,l)
%KMEANS Vector quantisation using K-means algorithm [X,ESQ,J]=(D,K,X0)
%
%  Inputs:
%
%    D(N,P)  contains N data vectors of dimension P
%    K       is number of centres required
%    X0(K,P) are the initial centres (optional)
%     
%      or alternatively
%
%    X0      gives the initialization method
%            'f'   pick K random elements of D as the initial centres [default]
%            'p'   randomly divide D into K sets and choose the centroids
%    L       gives max number of iterations (use 0 if you just want to calculate G and J)
%
%  Outputs:
%
%    X(K,P)  is output row vectors (omitted if L=0)
%    G       is mean square error
%    J(N)    indicates which centre each data vector belongs to
%    GG(L)   gives the mean square error at the start of each iteration (omitted if L=0)
%
% It is often a good idea to scale the input data so that it has equal variance in each
% dimension before calling KMEANS.

%  Originally based on a routine by Chuck Anderson, anderson@cs.colostate.edu, 1996


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

memsize=voicebox('memsize'); 
[n,p] = size(d);
nb=min(n,max(1,floor(memsize/(8*p*k))));    % block size for testing data points
nl=ceil(n/nb);                  % number of blocks
if nargin<4
    l=300;                  % very large max iteration count
    if nargin<3
        x0='f';             % use 'f' initialization mode
    end
end
if ischar(x0)
    if k<n
        if any(x0)=='p'                  % Initialize using a random partition
            ix=ceil(rand(1,n)*k);       % allocate to random clusters
            ix(rnsubset(k,n))=1:k;      % but force at least one point per cluster
            x=zeros(k,p);
            for i=1:k
                x(i,:)=mean(d(ix==i,:),1);
            end
        else                                % Forgy initialization: choose k random points [default] 
            x=d(rnsubset(k,n),:);         % sample k centres without replacement
        end
    else
        x=d(mod((1:k)-1,n)+1,:);    % just include all points several times
    end
else
    x=x0;
end
m=zeros(n,1);           % minimum distance to a centre
j=zeros(n,1);           % index of closest centre
gg=zeros(l,1);
wp=ones(1,p);
kk=1:p;
kk=kk(ones(n,1),:);
kk=kk(:);

if l>0
    for ll=1:l                 % loop until x==y causes a break
        
        % find closest centre to each data point [m(:),j(:)] = distance, index
        
        ix=1;
        jx=n-nl*nb;
        for il=1:nl
            jx=jx+nb;        % increment upper limit
            ii=ix:jx;
            z = disteusq(d(ii,:),x,'x');
            [m(ii),j(ii)] = min(z,[],2);
            ix=jx+1;
        end
        y = x;              % save old centre list
        
        % calculate new centres as the mean of their assigned data values (or zero for unused centres)
        
        nd=full(sparse(j,1,1,k,1));         % number of points allocated to each centre
        md=max(nd,1);                       % remove zeros
        jj=j(:,wp);
        x=full(sparse(jj(:),kk,d(:),k,p))./md(:,wp);    % calculate the new means 
        fx=find(nd==0);
        
        % if any centres are unused, assign them to data values that are not exactly on centres
        % choose randomly if there are more such points than needed
        
        if length(fx)
            q=find(m~=0);
            if length(q)<=length(fx)
                x(fx(1:length(q)),:)=d(q,:);
            else
                if length(fx)>1
                    [rr,ri]=sort(rand(length(q),1));
                    x(fx,:)=d(q(ri(1:length(fx))),:);
                else
                    x(fx,:) = d(q(ceil(rand(1)*length(q))),:);
                end
            end
        end
        
        % quit if the centres are unchanged
        
        gg(ll)=sum(m,1);
        if x==y
            break
        end
    end
    gg=gg(1:ll)/n;
%     ll % *** DEBUG ***
%     gg' % *** DEBUG ***
    g=gg(end);
else            % if l==0 then just calculate G and J (but rename as X and G)
    ix=1;
    jx=n-nl*nb;
    for il=1:nl
        jx=jx+nb;        % increment upper limit
        ii=ix:jx;
        z = disteusq(d(ii,:),x,'x');
        [m(ii),j(ii)] = min(z,[],2);
        ix=jx+1;
    end
    x=sum(m,1)/n;
    g=j;
end



function m = rnsubset(k,n)
%RNSUBSET choose k distinct random integers from 1:n M=(K,N)
%
%  Inputs:
%
%    K is number of disinct integers required from the range 1:N
%    N specifies the range - we must have K<=N
%
%  Outputs:
%
%    M(1,K) contains the output numbers

%      Copyright (C) Mike Brookes 2006
%      Version: $Id: rnsubset.m,v 1.1 2006/08/23 16:41:58 dmb Exp $
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if k>n
    error('rnsubset: k must be <= n');
end
% We use two algorithms according to the values of k and n
[f,e]=log2(n);
if k>0.03*n*(e-1)
[v,m]=sort(rand(1,n)); % for large k, just do a random permutation
else
    v=ceil(rand(1,k).*(n:-1:n-k+1));
    m=1:n;
    for i=1:k
        j=v(i)+i-1;
        x=m(i);
        m(i)=m(j);
        m(j)=x;
    end
end
m=m(1:k);



function y=voicebox(f,v)
%VOICEBOX  set global parameters for Voicebox functions Y=(FIELD,VAL)
%
%  Inputs:  F   is a field name
%           V   is a new value for the field
%
% Outputs:  Y   is set equal to the structure of parameters if the
%               f and v inputs are both present or both absent. If only
%               input f is specified, then y is set to the value of the
%               corresponding field or null if it doesn't exist.
%
% This routine contains default values for constants that are used by
% other functions in the VOICEBOX toolbox. Values in the first section below,
% entitled "System-dependent directory paths" should be set as follows:
%    PP.dir_temp     directory for storing temporary files
%    PP.dir_data     default directory to preappend to speech data file names 
%                    when the "d" option is specified in READWAV etc.
%    PP.shorten      location of SHORTEN executable. SHORTEN is a proprietary file compression
%                    algorithm that is used for some SPHERE-format files. READSPH
%                    will try to call an external decoder if it is asked to
%                    read such a compressed file.
%    PP.sfsbin       location of Speech Filing Sysytem binaries. If the "c" option
%                    is given to READSFS, it will try to create a requested item
%                    if it is not present in the SFS file. This parameter tells it
%                    where to find the SFS executables.
%    PP.sfssuffix    suffix for Speech Filing Sysytem binaries. READSFS uses this paremeter
%                    to create the name of an SFS executable (see PP.sfsbin above).
% Other values defined in this routine are the defaults for specific algorithm constants.
% If you want to change these, please refer to the individual routines for a fuller description.

% Bugs/Suggestions
%    (1)  Could allow a * at the end of F to act as a wildcard and return/print a part structure

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

persistent PP
if isempty(PP)
    
    % System-dependent directory paths and constants
    
    PP.dir_temp='F:\TEMP';                      % directory for storing temporary files
    PP.dir_data='E:\dmb\data\speech';           % default directory to preappend to speech data file names 
    PP.shorten='C:\bin\shorten.exe';            % location of shorten executable
    PP.sfsbin='F:\Program Files\SFS\Program';   % location of Speech Filing Sysytem binaries
    PP.sfssuffix='.exe';                        % suffix for Speech Filing Sysytem binaries
    PP.memsize=50e6;                            % Maximum amount of temporary memory to use (Bytes)
    
    % DYPSA glottal closure identifier
    
    PP.dy_cpfrac=0.3;           % presumed closed phase fraction of larynx cycle
    PP.dy_cproj=0.2;            % cost of projected candidate
    PP.dy_cspurt=-0.45;         % cost of a talkspurt
    PP.dy_dopsp=1;              % Use phase slope projection (1) or not (0)?
    PP.dy_ewdly=0.0008;         % window delay for energy cost function term [~ energy peak delay from closure] (sec)
    PP.dy_ewlen=0.003;          % window length for energy cost function term (sec)
    PP.dy_ewtaper=0.001;        % taper length for energy cost function window (sec)
    PP.dy_fwlen=0.00045;        % window length used to smooth group delay (sec)
    PP.dy_fxmax=500;            % max larynx frequency (Hz) 
    PP.dy_fxmin=50;             % min larynx frequency (Hz) 
    PP.dy_fxminf=60;            % min larynx frequency (Hz) [used for Frobenius norm only]
    PP.dy_gwlen=0.0030;         % group delay evaluation window length (sec)
    PP.dy_lpcdur=0.020;         % lpc analysis frame length (sec)
    PP.dy_lpcn=2;               % lpc additional poles
    PP.dy_lpcnf=0.001;          % lpc poles per Hz (1/Hz)
    PP.dy_lpcstep=0.010;        % lpc analysis step (sec)
    PP.dy_nbest=5;              % Number of NBest paths to keep
    PP.dy_preemph=50;           % pre-emphasis filter frequency (Hz) (to avoid preemphasis, make this very large)
    PP.dy_spitch=0.2;           % scale factor for pitch deviation cost
    PP.dy_wener=0.3;            % DP energy weighting
    PP.dy_wpitch=0.5;           % DP pitch weighting
    PP.dy_wslope=0.1;           % DP group delay slope weighting
    PP.dy_wxcorr=0.8;           % DP cross correlation weighting
    PP.dy_xwlen=0.01;           % cross-correlation length for waveform similarity (sec)
    
    % RAPT pitch tracker
    
    PP.rapt_f0min=50;           % Min F0 (Hz)
    PP.rapt_f0max=500;          % Max F0 (Hz)
    PP.rapt_tframe=0.01;        % frame size (s)
    PP.rapt_tlpw=0.005;         % low pass filter window size (s)
    PP.rapt_tcorw=0.0075;       % correlation window size (s)
    PP.rapt_candtr=0.3;         % minimum peak in NCCF
    PP.rapt_lagwt=0.3;          % linear lag taper factor
    PP.rapt_freqwt=0.02;        % cost factor for F0 change
    PP.rapt_vtranc=0.005;       % fixed voice-state transition cost
    PP.rapt_vtrac=0.5;          % delta amplitude modulated transition cost
    PP.rapt_vtrsc=0.5;          % delta spectrum modulated transition cost
    PP.rapt_vobias=0.0;         % bias to encourage voiced hypotheses
    PP.rapt_doublec=0.35;       % cost of exact doubling or halving
    PP.rapt_absnoise=0;         % absolute rms noise level
    PP.rapt_relnoise=2;         % rms noise level relative to noise floor
    PP.rapt_signoise=0.001;     % ratio of peak signal rms to noise floor (0.001 = 60dB)
    PP.rapt_ncands=20;          % max hypotheses at each frame
    PP.rapt_trms=0.03;                      % window length for rms measurement
    PP.rapt_dtrms=0.02;                     % window spacing for rms measurement
    PP.rapt_preemph=-7000;                  % s-plane position of preemphasis zero
    PP.rapt_nfullag=7;                      % number of full lags to try (must be odd)
    
    % now check some of the key values for validity
    
    if exist(PP.dir_temp)~=7        % check that temp directory exists
        PP.dir_temp = winenvar('temp');     % else use windows temp directory
    end
    
    if exist(PP.shorten)~=2        % check that shorten executable exists
        [fnp,fnn,fne]=fileparts(mfilename('fullpath'));
        PP.shorten=fullfile(fnp,'shorten.exe'); % next try local directory
        if exist(PP.shorten)~=2        % check if it exists in local directory
            PP.shorten='shorten.exe'; % finally assume it is on the search path
        end
    end
    
end
if nargin==0
    if nargout==0
        % list all fields
        nn=sort(fieldnames(PP));
        cnn=char(nn);
        fprintf('%d Voicebox parameters:\n',length(nn));
        
        for i=1:length(nn);
            if ischar(PP.(nn{i}))
                fmt='  %s = %s\n';
            else
                fmt='  %s = %g\n';
            end
            fprintf(fmt,cnn(i,:),PP.(nn{i}));   
        end
    else
        y=PP;
    end
elseif nargin==1
    if isfield(PP,f)
        y=PP.(f);
    else
        y=[];
    end
else
    if isfield(PP,f)
        PP.(f)=v;
        y=PP;
    else
        error(sprintf('''%s'' is not a valid voicebox field name',f));
    end
end



function d=winenvar(n)
%WINENVAR get windows environment variable [D]=(N)
%
% Inputs: N  name of environment variable (e.g. 'temp')
%
% Outputs: D  value of variable or [] is non-existant
%
% Notes: (1) This is WINDOWS specific and needs to be fixed to work on UNIX
%        (2) The search is case insensitive (like most of WINDOWS).
%
% Examples: (1) Open a temporary text file:
%               d=winenar('temp'); fid=fopen(fullfile(d,'temp.txt'),'wt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=['%',n,'%'];
[s,d]=system(['echo ',p]);
while d(end)<=' ';
    d(end)=[];
end
if strcmp(d,p)
    d=[];
end


function d=disteusq(x,y,mode,w)
%DISTEUSQ calculate euclidean, squared euclidean or mahanalobis distance D=(X,Y,MODE,W)
%
% Inputs: X,Y         Vector sets to be compared. Each row contains a data vector.
%                     X and Y must have the same number of columns.
%
%         MODE        Character string selecting the following options:
%                         'x'  Calculate the full distance matrix from every row of X to every row of Y
%                         'd'  Calculate only the distance between corresponding rows of X and Y
%                              The default is 'd' if X and Y have the same number of rows otherwise 'x'.
%                         's'  take the square-root of the result to give the euclidean distance.
%
%         W           Optional weighting matrix: the distance calculated is (x-y)*W*(x-y)'
%                     If W is a vector, then the matrix diag(W) is used.
%
% Output: D           If MODE='d' then D is a column vector with the same number of rows as the shorter of X and Y.
%                     If MODE='x' then D is a matrix with the same number of rows as X and the same number of columns as Y'.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nx,p]=size(x); ny=size(y,1);
if nargin<3 | isempty(mode) mode='0'; end
if any(mode=='d') | (mode~='x' & nx==ny)

    % Do pairwise distance calculation

    nx=min(nx,ny);
    z=x(1:nx,:)-y(1:nx,:);
    if nargin<4
        d=sum(z.*conj(z),2);
    elseif min(size(w))==1
        wv=w(:).';
        d=sum(z.*wv(ones(size(z,1),1),:).*conj(z),2);
    else
        d=sum(z*w.*conj(z),2);
    end
else
    
    % Calculate full distance matrix
    
    if p>1
        
        % x and y are matrices
        
        if nargin<4
            z=permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]);
            d=sum(z.*conj(z),3);
        else
            nxy=nx*ny;
            z=reshape(permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]),nxy,p);
            if min(size(w))==1
                wv=w(:).';
                d=reshape(sum(z.*wv(ones(nxy,1),:).*conj(z),2),nx,ny);
            else
                d=reshape(sum(z*w.*conj(z),2),nx,ny);
            end
        end
    else
        
        % x and y are vectors
        
        z=x(:,ones(1,ny))-y(:,ones(1,nx)).';
        if nargin<4
            d=z.*conj(z);
        else
            d=w*z.*conj(z);
        end
    end
end
if any(mode=='s')
    d=sqrt(d);
end

function [m, minIndx,p, DistHist]=vqsplit(X,L)
% Vector Quantization: K-Means Algorithm with Spliting Method for Training
% NOT TESTED FOR CODEBOOK SIZES OTHER THAN POWERS OF BASE 2, E.G. 256, 512, ETC
% (Saves output to a mat file (CBTEMP.MAT) after each itteration, so that if
% it is going too slow you can break it (CTRL+C) without losing your work
% so far.)
% [M, P, DH]=VQSPLIT(X,L)
% 
% or
% [M_New, P, DH]=VQSPLIT(X,M_Old)   In this case M_Old is a codebook and is
%                                   retrained on data X
% 
% inputs:
% X: a matrix each column of which is a data vector
% L: codebook size (preferably a power of 2 e.g. 16,32 256, 1024) (Never
% tested for other values!
% 
% Outputs:
% M: the codebook as the centroids of the clusters
% P: Weight of each cluster the number of its vectors divided by total
%       number of vectors
% DH: The total distortion history, a vector containing the overall
% distortion of each itteration
%
% Method:
% The mean vector is split to two. the model is trained on those two vectors
% until the distortion does not vary much, then those are split to two and
% so on. until the disired number of clusters is reached.
% Algorithm:
% 1. Find the Mean
% 2. Split each centroid to two
% 3. Assign Each Data to a centroid
% 4. Find the Centroids
% 5. Calculate The Total Distance
% 6. If the Distance has not changed much
%       if the number of Centroids is smaller than L2 Goto Step 2
%       else Goto 7
%    Else (the Distance has changed substantialy) Goto Step 3
% 7. If the number of Centroids is larger than L
%    Discard the Centroid with (highest distortion OR lowest population)
%    Goto 3
% 8. Calculate the Variances and Cluster Weights if required
% 9. End
%
% Esfandiar Zavarehei, Brunel University
% May-2006

e=.01; % X---> [X-e*X and X+e*X] Percentage for Spliting
eRed=0.75; % Rate of reduction of split size, e, after each spliting. i.e. e=e*eRed;
DT=.001; % The threshold in improvement in Distortion before terminating and spliting again
DTRed=0.75; % Rate of reduction of Improvement Threshold, DT, after each spliting
MinPop=0.10; % The population of each cluster should be at least 10 percent of its quota (N/LC)
             % Otherwise that codeword is replaced with another codeword


d=size(X,1); % Dimension
N=size(X,2); % Number of Data points
isFirstRound=1; % First Itteration after Spliting

if numel(L)==1
    M=mean(X,2); % Mean Vector
    CB=[M*(1+e) M*(1-e)]; % Split to two vectors
else
    CB=L; % If the codebook is passed to the function just train it
    L=size(CB,2);
    e=e*(eRed^fix(log2(L)));
    DT=DT*(DTRed^fix(log2(L)));
end

LC=size(CB,2); % Current size of the codebook

Iter=0;
Split=0;
IsThereABestCB=0;
maxIterInEachSize=50; % The maximum number of training itterations at each 
                      % codebook size (The codebook size starts from one 
                      % and increases thereafter)
EachSizeIterCounter=0;
while 1
    %Distance Calculation
    [minIndx, dst]=VQIndex(X,CB); % Find the closest codewords to each data vector

    ClusterD=zeros(1,LC);
    Population=zeros(1,LC);
    LowPop=[];
    % Find the Centroids (Mean of each Cluster)
    for i=1:LC
        Ind=find(minIndx==i);
        if length(Ind)<MinPop*N/LC % if a cluster has very low population just remember it
            LowPop=[LowPop i];
        else
            CB(:,i)=mean(X(:,Ind),2);
            Population(i)=length(Ind);
            ClusterD(i)=sum(dst(Ind));
        end        
    end
    if ~isempty(LowPop)
        [temp MaxInd]=maxn(Population,length(LowPop));
        CB(:,LowPop)=CB(:,MaxInd)*(1+e); % Replace low-population codewords with splits of high population codewords
        CB(:,MaxInd)=CB(:,MaxInd)*(1-e);
        
        %re-train
        [minIndx, dst]=VQIndex(X,CB);

        ClusterD=zeros(1,LC);
        Population=zeros(1,LC);
        
        for i=1:LC
            Ind=find(minIndx==i);
            if ~isempty(Ind)
                CB(:,i)=mean(X(:,Ind),2);
                Population(i)=length(Ind);
                ClusterD(i)=sum(dst(Ind));
            else %if no vector is close enough to this codeword, replace it with a random vector
                CB(:,i)=X(:,fix(rand*N)+1);
                disp('A random vector was assigned as a codeword.')
                isFirstRound=1;% At least another iteration is required
            end                
        end
    end
    Iter=Iter+1;
    if isFirstRound % First itteration after a split (dont exit)
        TotalDist=sum(ClusterD(~isnan(ClusterD)));
        DistHist(Iter)=TotalDist;
        PrevTotalDist=TotalDist;        
        isFirstRound=0;
    else
        TotalDist=sum(ClusterD(~isnan(ClusterD)));  
        DistHist(Iter)=TotalDist;
        PercentageImprovement=((PrevTotalDist-TotalDist)/PrevTotalDist);
        if PercentageImprovement>=DT %Improvement substantial
            PrevTotalDist=TotalDist; %Save Distortion of this iteration and continue training
            isFirstRound=0;
        else%Improvement NOT substantial (Saturation)
            EachSizeIterCounter=0;
            if LC>=L %Enough Codewords?
                if L==LC %Exact number of codewords
%                    disp(TotalDist)
                    break
                else % Kill one codeword at a time
                    [temp, Ind]=min(Population); % Eliminate low population codewords
                    NCB=zeros(d,LC-1);
                    NCB=CB(:,setxor(1:LC,Ind(1)));
                    CB=NCB;
                    LC=LC-1;
                    isFirstRound=1;
                end
            else %If not enough codewords yet, then Split more
                CB=[CB*(1+e) CB*(1-e)];
                e=eRed*e; %Split size reduction
                DT=DT*DTRed; %Improvement Threshold Reduction
                LC=size(CB,2);
                isFirstRound=1;
                Split=Split+1;
                IsThereABestCB=0; % As we just split this codebook, there is no best codebook at this size yet
%                disp(LC)
            end
        end
    end    
    if ~IsThereABestCB
        BestCB=CB;
        BestD=TotalDist;
        IsThereABestCB=1;
    else % If there is a best CB, check to see if the current one is better than that
        if TotalDist<BestD
            BestCB=CB;
            BestD=TotalDist;
        end
    end
    EachSizeIterCounter=EachSizeIterCounter+1;
    if EachSizeIterCounter>maxIterInEachSize % If too many itterations in this size, stop training this size
        EachSizeIterCounter=0;
        CB=BestCB; % choose the best codebook so far
        IsThereABestCB=0;
        if LC>=L %Enough Codewords?
            if L==LC %Exact number of codewords
%                disp(TotalDist)
                break
            else % Kill one codeword at a time
                [temp, Ind]=min(Population);
                NCB=zeros(d,LC-1);
                NCB=CB(:,setxor(1:LC,Ind(1)));
                CB=NCB;
                LC=LC-1;
                isFirstRound=1;
            end
        else %Split
            CB=[CB*(1+e) CB*(1-e)];
            e=eRed*e; %Split size reduction
            DT=DT*DTRed; %Improvement Threshold Reduction
            LC=size(CB,2);
            isFirstRound=1;
            Split=Split+1;
            IsThereABestCB=0;
%            disp(LC)
        end
    end        
%    disp(TotalDist);
    p=Population/N;
    save CBTemp CB p DistHist
end
m=CB;

p=Population/N;

%disp(['Iterations = ' num2str(Iter)])
%disp(['Split = ' num2str(Split)])

function [v, i]=maxn(x,n)
% [V, I]=MAXN(X,N)
% APPLY TO VECTORS ONLY!
% This function returns the N maximum values of vector X with their indices.
% V is a vector which has the maximum values, and I is the index matrix,
% i.e. the indices corresponding to the N maximum values in the vector X

if nargin<2
    [v, i]=max(x); %Only the first maximum (default n=1)
else
    n=min(length(x),n);
    [v, i]=sort(x);
    v=v(end:-1:end-n+1);
    i=i(end:-1:end-n+1);    
end
        
function [I, dst]=VQIndex(X,CB) 
% Distance function
% Returns the closest index of vectors in X to codewords in CB
% In other words:
% I is a vector. The length of I is equal to the number of columns in X.
% Each element of I is the index of closest codeword (column) of CB to
% coresponding column of X

L=size(CB,2);
N=size(X,2);
LNThreshold=64*10000;

if L*N<LNThreshold
    D=zeros(L,N);
    for i=1:L
        D(i,:)=sum((repmat(CB(:,i),1,N)-X).^2,1);
    end
    [dst I]=min(D);
else
    I=zeros(1,N);
    dst=I;
    for i=1:N
        D=sum((repmat(X(:,i),1,L)-CB).^2,1);
        [dst(i) I(i)]=min(D);
    end
end
    






