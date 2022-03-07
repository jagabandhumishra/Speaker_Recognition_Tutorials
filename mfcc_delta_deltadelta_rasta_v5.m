% In this function mfcc delta_mfcc and delta_delta_mfcc features are computated.
% In compution of delta features the first value is eliminated, other wise
% there will be a large diffrerence in first delta value. It will give three values.
% (1) mfcc features of dimension no of filters used
% (2) dlta_mfcc features of dimension no of filters used
% (3) dlta_delta_mfcc features of dimension no of filters used
% (4) mfcc appened with delta
% (5) mfcc appended with delta_delta features
% (5) mfcc appended with delta and delta_delta features


function [MFCC DMFCC DDMFCC ener_frame]=mfcc_delta_deltadelta_rasta_v5(d,Fs,Nomfccs,Nbands,Framesize,FrameShift,Del,DelDel,NDL)


d=d-mean(d); % mean substraction to make zero mean speech signal

MFCC1= mfcc_rasta_V1(d,Fs,'wintime',Framesize/1000,'hoptime',FrameShift/1000,'numcep',Nomfccs,'nbands', Nbands);

% MFCC1=MFCC1(2:end,:);

MFCC=MFCC1'; % this is taken for further computation 

%---------------- DYNAMIC FEATURES - ADDED BY PATI ON 14/01/10 ------------------

DMFCC=[];

DDMFCC=[];

if(Del==1 && DelDel==0)
    
    [DMFCC]=computedeltas_V2(MFCC,Nomfccs,NDL);
        
elseif(Del==1 && DelDel==1)
    
    [DMFCC DDMFCC]=computedeltas_V2(MFCC,Nomfccs,NDL);
    
end


%------------------------------------------ VAD - ADDED BY PKM ON 01/06/08 -----------------------------------------------------------------------
% energy_frame=energy_func(d,Framesize*(Fs/1000),FrameShift*(Fs/1000));
% lenergy_frame=length(energy_frame);

% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

% This voice activity detection is added by pati on 14/01/03.
% 
% The programm is taken from voicebox and modified by pati on 26/12/09

% This function avoids the for loop for energy frame detection

% energy_frame gives the location of the energy frames

% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[energy_frame,f]=ener_frames(d,Framesize*(Fs/1000),FrameShift*(Fs/1000)); 

% first and last four frames are discarded for faulty computation of delta coefficients 

energy_frame=energy_frame(find(energy_frame> 4 & energy_frame <=size(f,1)-4));
ener_frame=energy_frame;

MFCC=MFCC(energy_frame,:); % energy MFCC vectors

DMFCC=DMFCC(energy_frame,:); % energy DMFCC vectors

DDMFCC=DDMFCC(energy_frame,:); % energy DMFCC vectors

% --------------- ---------- DELTA FEATURE COMPUTATION --------------------------------------


function [D DD]=computedeltas_V2(x,L,NDL)

% X must be N X L  (N-> NO OF FRAMES  & L is DIMENSION OF THE FEATURE VECTOR) 
% L -> DIMENSION OF THE FEATURE VECTOR  (Eg: 13, if you are using 13 MFCC Coefs)

% Reference: Springer Hanbook of Speech Processing.

if(size(x,1)~=size(x,2))
if(size(x,1)==L)
    x=x';
end
end

D=finddelta2(x,NDL);          % DELTA
% D=D(NDL+1:end-NDL,:);  % this is added by prasanna sir to remove the first and last NDL frames

DD=finddelta2(D,NDL);         % DELTA - DELTA
% DD=DD(NDL+1:end-NDL,:); % this is added by prasanna sir to remove the first and last NDL frames


function y=finddelta2(x,NDL)

y=zeros(size(x));
N=size(x,1);

%------------------------------------------------------------------------
T=zeros(1,size(x,2));
for t=1:NDL
    T=T+(t.*x(1+t,:));
end
T=T./(sum(abs([1:NDL])));
y(1,:)=T;
%------------------------------------------------------------------------
for i=2:N
    
    if(i<=NDL)

             T=zeros(1,size(x,2));
             for t=-i+NDL-1:NDL
                 T=T+(t.*x(i+t,:));
             end
             T=T./(sum(abs([-i+NDL-1:NDL])));
             y(i,:)=T;
             
    elseif(i>(N-NDL))
        
             T=zeros(1,size(x,2));
             for t=-NDL:N-i
                 T=T+(t.*x(i+t,:));
             end
             T=T./(sum(abs([-NDL:N-i])));
             y(i,:)=T;
             
    else
        
             T=zeros(1,size(x,2));
             for t=-NDL:NDL
                 T=T+(t.*x(i+t,:));
             end
             T=T./(2*sum([1:NDL]));
             y(i,:)=T;
             
    end
        
end


function [loc,f,nef]=ener_frames(x,win,inc)
%ENFRAME split signal up into (overlapping) frames: one per row. F=(X,WIN,INC)
%
%	F = ENFRAME(X,LEN) splits the vector X(:) up into
%	frames. Each frame is of length LEN and occupies
%	one row of the output matrix. The last few frames of X
%	will be ignored if its length is not divisible by LEN.
%	It is an error if X is shorter than LEN.
%
%	F = ENFRAME(X,LEN,INC) has frames beginning at increments of INC
%	The centre of frame I is X((I-1)*INC+(LEN+1)/2) for I=1,2,...
%	The number of frames is fix((length(X)-LEN+INC)/INC)
%
%	F = ENFRAME(X,WINDOW) or ENFRAME(X,WINDOW,INC) multiplies
%	each frame by WINDOW(:)
%
% Example of frame-based processing:
%          INC=20       													% set frame increment
%          NW=INC*2     													% oversample by a factor of 2 (4 is also often used)
%          S=cos((0:NW*7)*6*pi/NW);								% example input signal
%          W=sqrt(hamming(NW+1)); W(end)=[];      % sqrt hamming window of period NW
%          F=enframe(S,W,INC);               			% split into frames
%          ... process frames ...
%          X=overlapadd(F,W,INC);           			% reconstitute the time waveform (omit "X=" to plot waveform)

%	   Copyright (C) Mike Brookes 1997
%      Version: $Id: enframe.m,v 1.6 2009/06/08 16:21:42 dmb Exp $
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
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nx=length(x(:));
nwin=length(win);
if (nwin == 1)
    len = win;
else
    len = nwin;
end
if (nargin < 3)
    inc = len;
end
nf = fix((nx-len+inc)/inc);
f=zeros(nf,len);
indf= inc*(0:(nf-1)).';
inds = (1:len);
f(:) = x(indf(:,ones(1,len))+inds(ones(nf,1),:));
if (nwin > 1)
    w = win(:)';
    f = f .* w(ones(nf,1),:);
end

% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

%      This is added by DDPATI on 26th december, 2009.

% The following codes give the normalized energy frames

% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

% Energy frames

% Energy frames are considered having more than 0.006*avg_energy

frameenergy=sum(f.*f,2); 

avg_frameenergy=mean(frameenergy);

loc=find(frameenergy > 0.06*avg_frameenergy); % location of the energy frames

energyframes=f(loc,:);

% Normalized frames

% Normalized frames are obtained by dividing the max abs value of

% resepetive fames

absmaximum=max(abs(energyframes),[],2); % Absolute maximum values of each frame

absmaximum=repmat(absmaximum,1,size(energyframes,2)); % Make it to dimension of energy frames for division

normenergyframes=energyframes./absmaximum; % normalized frames

nef=normenergyframes; % save as normalized energy frame (nef) as intialized


function [cepstra,aspectrum,pspectrum] = mfcc_rasta_V1(samples, sr, varargin)
%[cepstra,aspectrum,pspectrum] = melfcc(samples, sr[, opts ...])
%  Calculate Mel-frequency cepstral coefficients by:
%   - take the absolute value of the STFT
%   - warp to a Mel frequency scale
%   - take the DCT of the log-Mel-spectrum
%   - return the first <ncep> components
%  This version allows a lot of options to be controlled, as optional
%  'name', value pairs from the 3rd argument on: (defaults in parens)
%    'wintime' (0.025): window length in sec
%    'hoptime' (0.010): step between successive windows in sec
%    'numcep'     (13): number of cepstra to return
%    'lifterexp' (0.6): exponent for liftering; 0 = none; < 0 = HTK sin lifter
%    'sumpower'    (1): 1 = sum abs(fft)^2; 0 = sum abs(fft)
%    'preemph'  (0.97): apply pre-emphasis filter [1 -preemph] (0 = none)
%    'dither'      (0): 1 = add offset to spectrum as if dither noise
%    'minfreq'     (0): lowest band edge of mel filters (Hz)
%    'maxfreq'  (4000): highest band edge of mel filters (Hz)
%    'nbands'     (40): number of warped spectral bands to use
%    'bwidth'    (1.0): width of aud spec filters relative to default
%    'dcttype'     (2): type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac)
%    'fbtype'  ('mel'): frequency warp: 'mel','bark','htkmel','fcmel'
%    'usecmp'      (0): apply equal-loudness weighting and cube-root compr.
%    'modelorder'  (0): if > 0, fit a PLP model of this order
% The following non-default values nearly duplicate Malcolm Slaney's mfcc
% (i.e. melfcc(d,16000,opts...) =~= log(10)*2*mfcc(d*(2^17),16000) )
%       'wintime': 0.016
%     'lifterexp': 0
%       'minfreq': 133.33
%       'maxfreq': 6855.6
%      'sumpower': 0
% The following non-default values nearly duplicate HTK's MFCC
% (i.e. melfcc(d,16000,opts...) =~= 2*htkmelfcc(:,[13,[1:12]])'
%  where HTK config has PREEMCOEF = 0.97, NUMCHANS = 20, CEPLIFTER = 22,
%  NUMCEPS = 12, WINDOWSIZE = 250000.0, USEHAMMING = T, TARGETKIND = MFCC_0)
%     'lifterexp': -22
%        'nbands': 20
%       'maxfreq': 8000
%      'sumpower': 0
%        'fbtype': 'htkmel'
%       'dcttype': 3
% For more detail on reproducing other programs' outputs, see
% http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/mfccs.html
%
% 2005-04-19 dpwe@ee.columbia.edu after rastaplp.m.
% Uses Mark Paskin's process_options.m from KPMtools

if nargin < 2;   sr = 16000;    end

% Parse out the optional arguments
[wintime, hoptime, numcep, lifterexp, sumpower, preemph, dither, ...
    minfreq, maxfreq, nbands, bwidth, dcttype, fbtype, usecmp, modelorder] = ...
    process_options(varargin, 'wintime', 0.025, 'hoptime', 0.010, ...
    'numcep', 13, 'lifterexp', 0.6, 'sumpower', 1, 'preemph', 0.97, ...
    'dither', 0, 'minfreq', 0, 'maxfreq', 4000, ...
    'nbands', 40, 'bwidth', 1.0, 'dcttype', 2, ...
    'fbtype', 'mel', 'usecmp', 0, 'modelorder', 0);

if preemph ~= 0
    samples = filter([1 -preemph], 1, samples);
end

% Compute FFT power spectrum
pspectrum = powspec(samples, sr, wintime, hoptime, dither);

aspectrum = audspec(pspectrum, sr, nbands, fbtype, minfreq, maxfreq, sumpower, bwidth);

if (usecmp)
    % PLP-like weighting/compression
    aspectrum = postaud(aspectrum, maxfreq, fbtype);
end

if modelorder > 0

    if (dcttype ~= 1)
        disp(['warning: plp cepstra are implicitly dcttype 1 (not ', num2str(dcttype), ')']);
    end

    % LPC analysis
    lpcas = dolpc(aspectrum, modelorder);

    % convert lpc to cepstra
    cepstra = lpc2cep(lpcas, numcep);

    % Return the auditory spectrum corresponding to the cepstra?
    %  aspectrum = lpc2spec(lpcas, nbands);
    % else return the aspectrum that the cepstra are based on, prior to PLP

else

    % Convert to cepstra via DCT
    cepstra = spec2cep(aspectrum, numcep, dcttype);

end

cepstra = lifter(cepstra, lifterexp);

function y = powspec(x, sr, wintime, steptime, dither)
%y = powspec(x, sr, wintime, steptime, sumlin, dither)
%
% compute the powerspectrum of the input signal.
% basically outputs a power spectrogram
%
% each column represents a power spectrum for a given frame
% each row represents a frequency
%
% default values:
% sr = 8000Hz
% wintime = 25ms (200 samps)
% steptime = 10ms (80 samps)
% which means use 256 point fft
% hamming window

% for sr = 8000
%NFFT = 256;
%NOVERLAP = 120;
%SAMPRATE = 8000;
%WINDOW = hamming(200);

if nargin < 2
    sr = 8000;
end
if nargin < 3
    wintime = 0.025;
end
if nargin < 4
    steptime = 0.010;
end
if nargin < 5
    dither = 1;
end

winpts = round(wintime*sr);
steppts = round(steptime*sr);

NFFT = 2^(ceil(log(winpts)/log(2)));
WINDOW = hamming(winpts);
NOVERLAP = winpts - steppts;
SAMPRATE = sr;

% Values coming out of rasta treat samples as integers,
% not range -1..1, hence scale up here to match (approx)
y = abs(specgram(x*32768,NFFT,SAMPRATE,WINDOW,NOVERLAP)).^2;

% imagine we had random dither that had a variance of 1 sample
% step and a white spectrum.  That's like (in expectation, anyway)
% adding a constant value to every bin (to avoid digital zero)
if (dither)
    y = y + winpts;
end
% ignoring the hamming window, total power would be = #pts
% I think this doesn't quite make sense, but it's what rasta/powspec.c does

% that's all she wrote


function [aspectrum,wts] = audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth)
%[aspectrum,wts] = audspec(pspectrum, sr, nfilts, fbtype, minfreq, maxfreq, sumpower, bwidth)
%
% perform critical band analysis (see PLP)
% takes power spectrogram as input

if nargin < 2;  sr = 16000;                          end
if nargin < 3;  nfilts = ceil(hz2bark(sr/2))+1;      end
if nargin < 4;  fbtype = 'bark';  end
if nargin < 5;  minfreq = 0;    end
if nargin < 6;  maxfreq = sr/2; end
if nargin < 7;  sumpower = 1;   end
if nargin < 8;  bwidth = 1.0;   end

[nfreqs,nframes] = size(pspectrum);

nfft = (nfreqs-1)*2;

if strcmp(fbtype, 'bark')
    wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
elseif strcmp(fbtype, 'mel')
    wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
elseif strcmp(fbtype, 'htkmel')
    wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1);
elseif strcmp(fbtype, 'fcmel')
    wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0);
else
    disp(['fbtype ', fbtype, ' not recognized']);
    error;
end

wts = wts(:, 1:nfreqs);

% Integrate FFT bins into Mel bins, in abs or abs^2 domains:
if (sumpower)
    aspectrum = wts * pspectrum;
else
    aspectrum = (wts * sqrt(pspectrum)).^2;
end



function [y,eql] = postaud(x,fmax,fbtype,broaden)
%y = postaud(x, fmax, fbtype)
%
% do loudness equalization and cube root compression
% x = critical band filters
% rows = critical bands
% cols = frames

if nargin < 3
    fbtype = 'bark';
end
if nargin < 4
    % By default, don't add extra flanking bands
    broaden = 0;
end


[nbands,nframes] = size(x);

% equal loundness weights stolen from rasta code
%eql = [0.000479 0.005949 0.021117 0.044806 0.073345 0.104417 0.137717 ...
%      0.174255 0.215590 0.263260 0.318302 0.380844 0.449798 0.522813
%      0.596597];

% Include frequency points at extremes, discard later
nfpts = nbands+2*broaden;

if strcmp(fbtype, 'bark')
    bandcfhz = bark2hz(linspace(0, hz2bark(fmax), nfpts));
elseif strcmp(fbtype, 'mel')
    bandcfhz = mel2hz(linspace(0, hz2mel(fmax), nfpts));
elseif strcmp(fbtype, 'htkmel') || strcmp(fbtype, 'fcmel')
    bandcfhz = mel2hz(linspace(0, hz2mel(fmax,1), nfpts),1);
else
    disp(['unknown fbtype', fbtype]);
    error;
end

% Remove extremal bands (the ones that will be duplicated)
bandcfhz = bandcfhz((1+broaden):(nfpts-broaden));

% Hynek's magic equal-loudness-curve formula
fsq = bandcfhz.^2;
ftmp = fsq + 1.6e5;
eql = ((fsq./ftmp).^2) .* ((fsq + 1.44e6)./(fsq + 9.61e6));

% weight the critical bands
z = repmat(eql',1,nframes).*x;

% cube root compress
z = z .^ (.33);

% replicate first and last band (because they are unreliable as calculated)
if (broaden)
    y = z([1,1:nbands,nbands],:);
else
    y = z([2,2:(nbands-1),nbands-1],:);
end
%y = z([1,1:nbands-2,nbands-2],:);



function y = dolpc(x,modelorder)
%y = dolpc(x,modelorder)
%
% compute autoregressive model from spectral magnitude samples
%
% rows(x) = critical band
% col(x) = frame
%
% row(y) = lpc a_i coeffs, scaled by gain
% col(y) = frame
%
% modelorder is order of model, defaults to 8
% 2003-04-12 dpwe@ee.columbia.edu after shire@icsi.berkeley.edu

[nbands,nframes] = size(x);

if nargin < 2
    modelorder = 8;
end

% Calculate autocorrelation
r = real(ifft([x;x([(nbands-1):-1:2],:)]));
% First half only
r = r(1:nbands,:);

% Find LPC coeffs by durbin
[y,e] = levinson(r, modelorder);

% Normalize each poly by gain
y = y'./repmat(e',(modelorder+1),1);


% PROCESS_OPTIONS - Processes options passed to a Matlab function.
%                   This function provides a simple means of
%                   parsing attribute-value options.  Each option is
%                   named by a unique string and is given a default
%                   value.
%
% Usage:  [var1, var2, ..., varn[, unused]] = ...
%           process_options(args, ...
%                           str1, def1, str2, def2, ..., strn, defn)
%
% Arguments:
%            args            - a cell array of input arguments, such
%                              as that provided by VARARGIN.  Its contents
%                              should alternate between strings and
%                              values.
%            str1, ..., strn - Strings that are associated with a
%                              particular variable
%            def1, ..., defn - Default values returned if no option
%                              is supplied
%
% Returns:
%            var1, ..., varn - values to be assigned to variables
%            unused          - an optional cell array of those
%                              string-value pairs that were unused;
%                              if this is not supplied, then a
%                              warning will be issued for each
%                              option in args that lacked a match.
%
% Examples:
%
% Suppose we wish to define a Matlab function 'func' that has
% required parameters x and y, and optional arguments 'u' and 'v'.
% With the definition
%
%   function y = func(x, y, varargin)
%
%     [u, v] = process_options(varargin, 'u', 0, 'v', 1);
%
% calling func(0, 1, 'v', 2) will assign 0 to x, 1 to y, 0 to u, and 2
% to v.  The parameter names are insensitive to case; calling
% func(0, 1, 'V', 2) has the same effect.  The function call
%
%   func(0, 1, 'u', 5, 'z', 2);
%
% will result in u having the value 5 and v having value 1, but
% will issue a warning that the 'z' option has not been used.  On
% the other hand, if func is defined as
%
%   function y = func(x, y, varargin)
%
%     [u, v, unused_args] = process_options(varargin, 'u', 0, 'v', 1);
%
% then the call func(0, 1, 'u', 5, 'z', 2) will yield no warning,
% and unused_args will have the value {'z', 2}.  This behaviour is
% useful for functions with options that invoke other functions
% with options; all options can be passed to the outer function and
% its unprocessed arguments can be passed to the inner function.

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varargout] = process_options(args, varargin)

% Check the number of input arguments
n = length(varargin);
if (mod(n, 2))
    error('Each option must be a string/value pair.');
end

% Check the number of supplied output arguments
if (nargout < (n / 2))
    error('Insufficient number of output arguments given');
elseif (nargout == (n / 2))
    warn = 1;
    nout = n / 2;
else
    warn = 0;
    nout = n / 2 + 1;
end

% Set outputs to be defaults
varargout = cell(1, nout);
for i=2:2:n
    varargout{i/2} = varargin{i};
end

% Now process all arguments
nunused = 0;
for i=1:2:length(args)
    found = 0;
    for j=1:2:n
        if strcmpi(args{i}, varargin{j})
            varargout{(j + 1)/2} = args{i + 1};
            found = 1;
            break;
        end
    end
    if (~found)
        if (warn)
            warning(sprintf('Option ''%s'' not used.', args{i}));
            args{i}
        else
            nunused = nunused + 1;
            unused{2 * nunused - 1} = args{i};
            unused{2 * nunused} = args{i + 1};
        end
    end
end

% Assign the unused arguments
if (~warn)
    if (nunused)
        varargout{nout} = unused;
    else
        varargout{nout} = cell(0);
    end
end


function [wts,binfrqs] = fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel, constamp)
% wts = fft2melmx(nfft, sr, nfilts, width, minfrq, maxfrq, htkmel, constamp)
%      Generate a matrix of weights to combine FFT bins into Mel
%      bins.  nfft defines the source FFT size at sampling rate sr.
%      Optional nfilts specifies the number of output bands required
%      (else one per bark), and width is the constant width of each
%      band relative to standard Mel (default 1).
%      While wts has nfft columns, the second half are all zero.
%      Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
%      minfrq is the frequency (in Hz) of the lowest band edge;
%      default is 0, but 133.33 is a common standard (to skip LF).
%      maxfrq is frequency in Hz of upper edge; default sr/2.
%      You can exactly duplicate the mel matrix in Slaney's mfcc.m
%      as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
%      htkmel=1 means use HTK's version of the mel curve, not Slaney's.
%      constamp=1 means make integration windows peak at 1, not sum to 1.
% 2004-09-05  dpwe@ee.columbia.edu  based on fft2barkmx

if nargin < 2;     sr = 8000;      end
if nargin < 3;     nfilts = 40;    end
if nargin < 4;     width = 1.0;    end
if nargin < 5;     minfrq = 0;     end  % default bottom edge at 0
if nargin < 6;     maxfrq = sr/2;  end  % default top edge at nyquist
if nargin < 7;     htkmel = 0;     end
if nargin < 8;     constamp = 0;   end


wts = zeros(nfilts, nfft);

% Center freqs of each FFT bin
fftfrqs = [0:nfft-1]/nfft*sr;

% 'Center freqs' of mel bands - uniformly spaced between limits
minmel = hz2mel(minfrq, htkmel);
maxmel = hz2mel(maxfrq, htkmel);
binfrqs = mel2hz(minmel+[0:(nfilts+1)]/(nfilts+1)*(maxmel-minmel), htkmel);

binbin = round(binfrqs/sr*(nfft-1));

for i = 1:nfilts
    %  fs = mel2hz(i + [-1 0 1], htkmel);
    fs = binfrqs(i+[0 1 2]);
    % scale by width
    fs = fs(2)+width*(fs - fs(2));
    % lower and upper slopes for all bins
    loslope = (fftfrqs - fs(1))/(fs(2) - fs(1));
    hislope = (fs(3) - fftfrqs)/(fs(3) - fs(2));
    % .. then intersect them with each other and zero
    %  wts(i,:) = 2/(fs(3)-fs(1))*max(0,min(loslope, hislope));
    wts(i,:) = max(0,min(loslope, hislope));

    % actual algo and weighting in feacalc (more or less)
    %  wts(i,:) = 0;
    %  ww = binbin(i+2)-binbin(i);
    %  usl = binbin(i+1)-binbin(i);
    %  wts(i,1+binbin(i)+[1:usl]) = 2/ww * [1:usl]/usl;
    %  dsl = binbin(i+2)-binbin(i+1);
    %  wts(i,1+binbin(i+1)+[1:(dsl-1)]) = 2/ww * [(dsl-1):-1:1]/dsl;
    % need to disable weighting below if you use this one

end

if (constamp == 0)
    % Slaney-style mel is scaled to be approx constant E per channel
    wts = diag(2./(binfrqs(2+[1:nfilts])-binfrqs(1:nfilts)))*wts;
end

% Make sure 2nd half of FFT is zero
wts(:,(nfft/2+1):nfft) = 0;
% seems like a good idea to avoid aliasing


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = mel2hz(z, htk)
%   f = mel2hz(z, htk)
%   Convert 'mel scale' frequencies into Hz
%   Optional htk = 1 means use the HTK formula
%   else use the formula from Slaney's mfcc.m
% 2005-04-19 dpwe@ee.columbia.edu

if nargin < 2
    htk = 0;
end

if htk == 1
    f = 700*(10.^(z/2595)-1);
else

    f_0 = 0; % 133.33333;
    f_sp = 200/3; % 66.66667;
    brkfrq = 1000;
    brkpt  = (brkfrq - f_0)/f_sp;  % starting mel value for log region
    logstep = exp(log(6.4)/27); % the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = (z < brkpt);

    f = 0*z;

    % fill in parts separately
    f(linpts) = f_0 + f_sp*z(linpts);
    f(~linpts) = brkfrq*exp(log(logstep)*(z(~linpts)-brkpt));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = hz2mel(f,htk)
%  z = hz2mel(f,htk)
%  Convert frequencies f (in Hz) to mel 'scale'.
%  Optional htk = 1 uses the mel axis defined in the HTKBook
%  otherwise use Slaney's formula
% 2005-04-19 dpwe@ee.columbia.edu

if nargin < 2
    htk = 0;
end

if htk == 1
    z = 2595 * log10(1+f/700);
else
    % Mel fn to match Slaney's Auditory Toolbox mfcc.m

    f_0 = 0; % 133.33333;
    f_sp = 200/3; % 66.66667;
    brkfrq = 1000;
    brkpt  = (brkfrq - f_0)/f_sp;  % starting mel value for log region
    logstep = exp(log(6.4)/27); % the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

    linpts = (f < brkfrq);

    z = 0*f;

    % fill in parts separately
    z(linpts) = (f(linpts) - f_0)/f_sp;
    z(~linpts) = brkpt+(log(f(~linpts)/brkfrq))./log(logstep);

end

function [cep,dctm] = spec2cep(spec, ncep, type)
% [cep,dctm] = spec2cep(spec, ncep, type)
%     Calculate cepstra from spectral samples (in columns of spec)
%     Return ncep cepstral rows (defaults to 9)
%     This one does type II dct, or type I if type is specified as 1
%     dctm returns the DCT matrix that spec was multiplied by to give cep.
% 2005-04-19 dpwe@ee.columbia.edu  for mfcc_dpwe

if nargin < 2;   ncep = 13;   end
if nargin < 3;   type = 2;   end   % type of DCT

[nrow, ncol] = size(spec);

% Make the DCT matrix
dctm = zeros(ncep, nrow);
if type == 2 || type == 3
    % this is the orthogonal one, the one you want
    for i = 1:ncep
        dctm(i,:) = cos((i-1)*[1:2:(2*nrow-1)]/(2*nrow)*pi) * sqrt(2/nrow);
    end
    if type == 2
        % make it unitary! (but not for HTK type 3)
        dctm(1,:) = dctm(1,:)/sqrt(2);
    end
elseif type == 4 % type 1 with implicit repeating of first, last bins
    % Deep in the heart of the rasta/feacalc code, there is the logic
    % that the first and last auditory bands extend beyond the edge of
    % the actual spectra, and they are thus copied from their neighbors.
    % Normally, we just ignore those bands and take the 19 in the middle,
    % but when feacalc calculates mfccs, it actually takes the cepstrum
    % over the spectrum *including* the repeated bins at each end.
    % Here, we simulate 'repeating' the bins and an nrow+2-length
    % spectrum by adding in extra DCT weight to the first and last
    % bins.
    for i = 1:ncep
        dctm(i,:) = cos((i-1)*[1:nrow]/(nrow+1)*pi) * 2;
        % Add in edge points at ends (includes fixup scale)
        dctm(i,1) = dctm(i,1) + 1;
        dctm(i,nrow) = dctm(i,nrow) + ((-1)^(i-1));
    end
    dctm = dctm / (2*(nrow+1));
else % dpwe type 1 - same as old spec2cep that expanded & used fft
    for i = 1:ncep
        dctm(i,:) = cos((i-1)*[0:(nrow-1)]/(nrow-1)*pi) * 2 / (2*(nrow-1));
    end
    % fixup 'non-repeated' points
    dctm(:,[1 nrow]) = dctm(:, [1 nrow])/2;
end

cep = dctm*log(spec);

function y = lifter(x, lift, invs)
% y = lifter(x, lift, invs)
%   Apply lifter to matrix of cepstra (one per column)
%   lift = exponent of x i^n liftering
%   or, as a negative integer, the length of HTK-style sin-curve liftering.
%   If inverse == 1 (default 0), undo the liftering.
% 2005-05-19 dpwe@ee.columbia.edu

if nargin < 2;   lift = 0.6; end   % liftering exponent
if nargin < 3;   invs = 0; end      % flag to undo liftering

[ncep, nfrm] = size(x);

if lift == 0
    y = x;
else

    if lift > 0
        if lift > 10
            disp(['Unlikely lift exponent of ', num2str(lift),' (did you mean -ve?)']);
        end
        liftwts = [1, ([1:(ncep-1)].^lift)];
    elseif lift < 0
        % Hack to support HTK liftering
        L = -lift;
        if (L ~= round(L))
            disp(['HTK liftering value ', num2str(L),' must be integer']);
        end
        liftwts = [1, (1+L/2*sin([1:(ncep-1)]*pi/L))];
    end

    if (invs)
        liftwts = 1./liftwts;
    end

    y = diag(liftwts)*x;

end

