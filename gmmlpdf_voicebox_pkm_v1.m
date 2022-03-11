function l=gmmlpdf_voicebox_pkm_v1(x,m,v,w)
%GMMPDF calculated the pdf of a mixture of gaussians p=(x,m,v,w)
%
% Inputs: n data values, k mixtures, p parameters
%
%     X(n,p)   Input data vectors, one per row.
%     M(k,p)   mixture means, one row per mixture.
%     V(k,p)   mixture variances, one row per mixture (singlton dimensions will be replicated as required)
%              or else V(p,p,k) for full mixture covariance matrixes           
%     W(k,1)   mixture weights, one per mixture. The weights will be normalized by their sum. [default: all equal]
%
% Outputs: (Note that M, V and W are omitted if L==0)
%
%     L(n,1)   log PDF values

%  Bugs/Suggestions
%     (1) Sort out full covariance maatrices
%     (2) Improve plotting
%     (3) Add an extra arument for plotting control

%      Copyright (C) Mike Brookes 2000-2006
%      Version: $Id: gmmlpdf.m,v 1.1 2006/09/04 20:36:30 dmb Exp $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,p]=size(x);
l=[];           % in case n=0
if nargout>0 || n>0
    x2=x.^2;            % need x^2 for variance calculation
    k=size(m,1);        % number of mixtures
    if size(m,2)~=p
        error('x and m must have the same number of columns');
    end
    if nargin<4
        if nargin<3
            v=1;
        end
        w=ones(k,1);
    end
    w=w/sum(w);         % normalize the weights
    sv=size(v);
    if length(sv)>2 || k==1 && p>1 && sv(1)==p     % full covariance matrices
        error('full covariance matrices not yet implemented');
    else                            % diagonal (or constant) covariance matrices
        if sv(1)==1
            v=v(ones(k,1),:);
        end
        if sv(2)==1
            v=v(:,ones(1,p));
        end
        
        % If data size is large then do calculations in chunks
        
        memsize=voicebox('memsize'); 
        nb=min(n,max(1,floor(memsize/(8*p*k))));    % chunk size for testing data points
        nl=ceil(n/nb);                  % number of chunks
        
        im=repmat(1:k,1,nb); im=im(:);
        
        lpx=zeros(n,1);             % log probability of each data point
        wk=ones(k,1);
        wnb=ones(1,nb);
        vi=v.^(-1);                 % calculate quantities that depend on the variances
        vm=sqrt(prod(vi,2)).*w;
        vi=-0.5*vi;
        
        % first do partial chunk
        
        jx=n-(nl-1)*nb;                % size of first chunk
        ii=1:jx;
        kk=repmat(ii,k,1);
        km=repmat(1:k,1,jx);
        py=reshape(sum((x(kk(:),:)-m(km(:),:)).^2.*vi(km(:),:),2),k,jx);
        mx=max(py,[],1);                % find normalizing factor for each data point to prevent underflow when using exp()
        px=exp(py-mx(wk,:)).*vm(:,ones(1,jx));  % find normalized probability of each mixture for each datapoint
               
        lpx(ii)=log(sum(px,1))+mx;
        ix=jx+1;
        
        for il=2:nl
            jx=jx+nb;        % increment upper limit
            ii=ix:jx;
            kk=repmat(ii,k,1);
            py=reshape(sum((x(kk(:),:)-m(im,:)).^2.*vi(im,:),2),k,nb);
            mx=max(py,[],1);                % find normalizing factor for each data point to prevent underflow when using exp()
            px=exp(py-mx(wk,:)).*vm(:,wnb);  % find normalized probability of each mixture for each datapoint
            lpx(ii)=log(sum(px,1))+mx;
            ix=jx+1;
        end
        l=lpx-0.5*p*log(2*pi);   % log of total probability of each data point
    end
end
if nargout==0                        % attempt to plot the result
    if p==1                            % one dimensional data          
        plot(x,l);
    end
end

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

%      Copyright (C) Mike Brookes 2003
%      Version: $Id: voicebox.m,v 1.14 2006/07/15 21:36:11 dmb Exp $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
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
%         [fnp,fnn,fne,fnv]=fileparts(mfilename('fullpath'));
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

%   Copyright (c) 2005 Mike Brookes,  mike.brookes@ic.ac.uk
%      Version: $Id: winenvar.m,v 1.1 2006/06/25 21:28:18 dmb Exp $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   ftp://prep.ai.mit.edu/pub/gnu/COPYING-2.0 or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=['%',n,'%'];
[s,d]=system(['echo ',p]);
while d(end)<=' ';
    d(end)=[];
end
if strcmp(d,p)
    d=[];
end