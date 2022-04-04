function generateMATLABFunction(obj,varargin)
%GENERATEMATLABFUNCTION Create MATLAB function compatible with C/C++ code
%generation
%
% generateMATLABFunction(afe) generates code and opens an untitled file
% containing a function named extractAudioFeatures.
%  The generated MATLAB function has the following signature:
%    featureVector = extractAudioFeatures(audioIn)
%  which is equivalent to:
%    featureVector = extract(afe,audioIn)
%
% generateMATLABFunction(afe,fileName) generates code and saves the
% resulting function to the file specified by fileName.
%
% generateMATLABFunction(...,IsStreaming=TF) specifies whether the
% function is intended for stream (single-frame) processing. You can use
% this syntax with either of the previous syntaxes. If TF is specified as
% true, the resulting function requires single-frame inputs of length
% numel(afe.Window). If individual feature extractors have state, the
% resulting function maintains the state between calls. If unspecified, TF
% defaults to false. The streaming function has the following signature:
%    featureVector = functionName(audioIn,varargin)
% where
%    - featureVector is returned as an M-by-N matrix, where M is the number
%      of features extracted and N is the number of channels. This is in
%      contrast to the output to the non-streaming mode, which is returned
%      as an L-by-M-by-N array, where L is the number of hops, M is the
%      number of feature vectors, and N is the number of channels.
%    - varargin can be the optional name-value pair 'Reset' and either true
%      or false. If you call the function with 'Reset' set to true, then it
%      clears any state before calculating and returning the feature vector.
%
% % EXAMPLE: Generate MATLAB code to untitled file.
%   afe = audioFeatureExtractor(pitch=true,barkSpectrum=true);
%   generateMATLABFunction(afe)
%
% % EXAMPLE: Generate and save MATLAB code to named file.
%   afe = audioFeatureExtractor(mfcc=true,spectralCentroid=true,harmonicRatio=true);
%   generateMATLABFunction(afe,'extractAudioFeatures')
%
% % EXAMPLE: Generate and save MATLAB code to named file that is compatible
% %          with streaming. The generated function maintains state between
% %          calls.
%   afe = audioFeatureExtractor(mfcc=true,mfccDelta=true);
%   generateMATLABFunction(afe,'extractAudioFeatures',IsStreaming=true)
%
%
% See also AUDIOFEATUREEXTRACTOR, DSP.ASYNCBUFFER, CODEGEN

% Copyright 2020-2021 The MathWorks, Inc.

if ~obj.pIsInitialized
    prototype = zeros(0,'double');
    setup(obj,prototype)
end
config = obj.pPipelineParameters;


coder.internal.errorIf(config.FeaturesPerHop==0,'audio:audioFeatureExtractor:InvalidConfigurationNoFeatures', ...
    'audioFeatureExtractor')

precision = 32;

UsesState = iCheckUsesState(obj);
UsesConfig = iCheckUsesConfig(obj);
UsesWindow = iCheckUsesWindow(obj);

settings = struct('IsStreaming',false);
settings = matlabshared.fusionutils.internal.setProperties(settings,nargin-1,varargin{:},'nameOrPath');
validateattributes(settings.IsStreaming,{'numeric','logical'},{'scalar'},'generateMATLABFunction','TF')
IsStreaming = settings.IsStreaming;

% Error if unsupported configuration
pitchParams = getFeatureExtractorParams(obj,'pitch');
coder.internal.errorIf(IsStreaming && obj.pitch && pitchParams.MedianFilterLength ~= 1, ...
    'audio:audioFeatureExtractor:UnsupportedPitchParamMedianFilter')
coder.internal.errorIf(IsStreaming && obj.pitch && strcmpi(pitchParams.Method,'SRH'), ...
    'audio:audioFeatureExtractor:UnsupportedPitchParamSRHMethod')

% Determine which buffering/windowing is needed
[UseBuffer,PreBufferPitch,UseBuffer3DWindow,UseBuffer3DNoWindow,UseBuffer2DWindow] = iCheckBufferUsage(obj,IsStreaming);

% Set function name and check validity
if isfield(settings,'nameOrPath')
    nameOrPath = settings.nameOrPath;
    usesDefaultName = false;
else
    nameOrPath = 'extractAudioFeatures';
    usesDefaultName = true;
end
nameOrPath = convertStringsToChars(nameOrPath);
[filePath,functionName] = fileparts(nameOrPath);
coder.internal.errorIf(~isvarname(functionName), ...
    'audio:audioFeatureExtractor:InvalidFunctionName')

% Create function generator
functionGenerator = sigutils.internal.emission.MatlabFunctionGenerator;
functionGenerator.AlignEndOfLineComments = false;
functionGenerator.Name = functionName;
functionGenerator.Path = filePath;
functionGenerator.InputArgs = {'x'};
functionGenerator.OutputArgs = {'featureVector'};

% Add help
iCreateFunctionHelp(obj,functionGenerator,IsStreaming,UsesState,functionName)

% Add input parsing
addCode(functionGenerator,'dataType = underlyingType(x);')
if IsStreaming
    addCode(functionGenerator,'numChannels = size(x,2);')
else
    addCode(functionGenerator,'[numSamples,numChannels] = size(x);')
end

% Add config and state initialization
iAddConfigAndStateInitialization(obj,functionGenerator,IsStreaming,UsesState,UsesConfig)

% Add preallocation
iAddPreallocation(functionGenerator,IsStreaming,UsesWindow)

% Add buffering
iAddBuffering(functionGenerator,UseBuffer,UseBuffer3DWindow,UseBuffer3DNoWindow,UseBuffer2DWindow)

% Add feature extraction
iAddSpectrum(obj,functionGenerator,IsStreaming)
iAddLinearSpectrum(obj,functionGenerator,IsStreaming)
iAddMelSpectrum(obj,functionGenerator,IsStreaming)
iAddBarkSpectrum(obj,functionGenerator,IsStreaming)
iAddERBSpectrum(obj,functionGenerator,IsStreaming)
iAddMFCC(obj,functionGenerator,IsStreaming)
iAddGTCC(obj,functionGenerator,IsStreaming)
if obj.pExtractSpectralDescriptor
    addCode(functionGenerator)
    addCode(functionGenerator,'% Spectral descriptors')
    iAddSpectralMoments(obj,functionGenerator,IsStreaming) % Centroid, Spread, Skewness, Kurtosis
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralCrest')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralDecrease')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralEntropy')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralFlatness')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralFlux')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralRolloffPoint')
    iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,'spectralSlope')
end
iAddPeriodicityMetrics(obj,functionGenerator,IsStreaming,UsesState,UsesWindow,UseBuffer,PreBufferPitch)
iAddEnergyMetrics(obj,functionGenerator,IsStreaming)

% Add subfunctions
iAddSubfunction_getProps(obj,functionGenerator,IsStreaming,UsesWindow,precision);
iAddSubfunction_getConfig(obj,functionGenerator,IsStreaming,UsesConfig,precision);
iAddSubfunction_getState(obj,functionGenerator,IsStreaming,UsesState);
iAddSubfunction_reset(obj,functionGenerator,IsStreaming,UsesState);

% Create the function
if usesDefaultName
    fb = getFileBuffer(functionGenerator);
    dsp.util.sendToEditor(char(fb),false);
else
    openFileForEdit = true;
    writeFile(functionGenerator,openFileForEdit);
end

end

function outIndex = getOutIdx(IsStreaming,featureName)
if IsStreaming
    outIndex = sprintf('featureVector(outputIndex.%s,:)',featureName);
else
    outIndex = sprintf('featureVector(:,outputIndex.%s,:)',featureName);
end
end

function addAudSpec(functionGenerator,audSpec,audSpecType,audSpecIsUsed,audSpecIsOutput,IsStreaming)
if strcmpi(audSpecType,"Power")
    audSpecInput = 'Zpower';
else
    audSpecInput = 'Zmagnitude';
end
if IsStreaming %STREAMING
    if audSpecIsUsed && audSpecIsOutput
        addCode(functionGenerator,'%s = config.%s.FilterBank*%s;',audSpec,audSpec,audSpecInput)
        addCode(functionGenerator,'featureVector(outputIndex.%s,:) = %s;',audSpec,audSpec)
        addCode(functionGenerator,'%s = reshape(%s,[],1,numChannels);',audSpec,audSpec)
    elseif ~audSpecIsUsed && audSpecIsOutput
        addCode(functionGenerator,'%s = config.%s.FilterBank*%s;',getOutIdx(IsStreaming,audSpec),audSpec,audSpecInput)
    elseif audSpecIsUsed && ~audSpecIsOutput
        addCode(functionGenerator,'%s = reshape(config.%s.FilterBank*%s,[],1,numChannels);',audSpec,audSpec,audSpecInput)
    end
else %BATCH
    addCode(functionGenerator,'%s = reshape(config.%s.FilterBank*%s,[],numHops,numChannels);',audSpec,audSpec,audSpecInput)
    if (audSpecIsUsed && audSpecIsOutput) || (~audSpecIsUsed && audSpecIsOutput)
        addCode(functionGenerator,'%s = permute(%s,[2,1,3]);',getOutIdx(IsStreaming,audSpec),audSpec)
    end
end
end

function idxDataType = getIndexingClass(amax)
if amax <= intmax('uint8')
    idxDataType = 'uint8';
elseif amax <= intmax('uint16')
    idxDataType = 'uint16';
else
    idxDataType = 'uint32';
end
end

function iAddConfigAndStateInitialization(obj,functionGenerator,IsStreaming,UsesState,UsesConfig)
addCode(functionGenerator)
addCode(functionGenerator,'props = coder.const(getProps(dataType));')
if ~IsStreaming || ~UsesState
    addCode(functionGenerator)
    if ~UsesConfig
        addCode(functionGenerator,'persistent outputIndex')
        addCode(functionGenerator,'if isempty(outputIndex)')
        addCode(functionGenerator,'outputIndex = coder.const(getConfig);')
    elseif (obj.linearSpectrum && numel(fields(obj.pOutputIndexReduced))==1) && ~obj.pUseWindowNormalizationPower && ~obj.pUseWindowNormalizationMagnitude
        addCode(functionGenerator,'persistent config outputIndex')
        addCode(functionGenerator,'if isempty(outputIndex)')
        addCode(functionGenerator,'[config, outputIndex] = coder.const(@getConfig,dataType);')
    else
        addCode(functionGenerator,'persistent config outputIndex')
        addCode(functionGenerator,'if isempty(outputIndex)')
        addCode(functionGenerator,'[config, outputIndex] = coder.const(@getConfig,dataType,props);')
    end
    addCode(functionGenerator,'end')
else
    addCode(functionGenerator)
    if UsesConfig
        addCode(functionGenerator,'persistent config outputIndex state')
    else
        addCode(functionGenerator,'persistent outputIndex state')
    end
    addCode(functionGenerator,'if isempty(outputIndex)')
    if UsesConfig
        addCode(functionGenerator,'[config, outputIndex] = coder.const(@getConfig,dataType,props);')
    else
        addCode(functionGenerator,'outputIndex = coder.const(getConfig);')
    end
    addCode(functionGenerator,'state = getState(dataType,numChannels);')
    addCode(functionGenerator,'else')
    addCode(functionGenerator,'assert(state.NumChannels == numChannels)')
    addCode(functionGenerator,'end')
    addCode(functionGenerator,'if nargin==3')
    addCode(functionGenerator,'if strcmpi(varargin{1},"Reset") && varargin{2}')
    addCode(functionGenerator,'state = reset(state);')
    addCode(functionGenerator,'end')
    addCode(functionGenerator,'end')
end
end

function iAddPreallocation(functionGenerator,IsStreaming,UsesWindow)
addCode(functionGenerator)
addCode(functionGenerator,'% Preallocate feature vector')
if IsStreaming
    addCode(functionGenerator,'featureVector = coder.nullcopy(zeros(props.NumFeatures,numChannels,dataType));');
else
    if UsesWindow
        addCode(functionGenerator,'numHops = floor((numSamples-numel(props.Window))/(numel(props.Window) - props.OverlapLength)) + 1;');
    else
        addCode(functionGenerator,'numHops = floor((numSamples-props.WindowLength)/(props.WindowLength - props.OverlapLength)) + 1;');
    end
    addCode(functionGenerator,'featureVector = coder.nullcopy(zeros(numHops,props.NumFeatures,numChannels,dataType));');
end
end

function iAddBuffering(functionGenerator,UseBuffer,UseBuffer3DWindow,UseBuffer3DNoWindow,UseBuffer2DWindow)
if UseBuffer
    addCode(functionGenerator)
    addCode(functionGenerator,'% Buffer signal')
    addCode(functionGenerator,'windowLength = numel(props.Window);')
    addCode(functionGenerator,'hopLength = windowLength - props.OverlapLength;')
    addCode(functionGenerator,'numHops = floor((numSamples-windowLength)/hopLength) + 1;');
    addCode(functionGenerator,'xb2 = zeros(windowLength,numHops*numChannels,"like",x);');
    addCode(functionGenerator,'for channel = 1:numChannels');
    addCode(functionGenerator,'for hop = 1:numHops');
    addCode(functionGenerator,'for k = 1:windowLength');
    addCode(functionGenerator,'xb2(k,hop+(channel-1)*numHops) = x(k+hopLength*(hop-1),channel);');
    addCode(functionGenerator,'end');
    addCode(functionGenerator,'end');
    addCode(functionGenerator,'end');
    if UseBuffer3DNoWindow
        addCode(functionGenerator,'xb3 = reshape(xb2,windowLength,[],numChannels);')
    end
    if UseBuffer2DWindow || UseBuffer3DWindow
        addCode(functionGenerator,'xb2w = bsxfun(@times,xb2,props.Window);')
        if UseBuffer3DWindow
            addCode(functionGenerator,'xb3w = reshape(xb2w,windowLength,[],numChannels);');
        end
    end
end
end
function iAddSpectrum(obj,functionGenerator,IsStreaming)
if obj.pUseSpectrum
    addCode(functionGenerator)
    if IsStreaming
        addCode(functionGenerator,'% Fourier transform')
        addCode(functionGenerator,'Y = fft(bsxfun(@times,x,props.Window),props.FFTLength);');
        addCode(functionGenerator,'Z = Y(config.OneSidedSpectrumBins,:);');
    else
        addCode(functionGenerator,'% Short-time Fourier transform')
        addCode(functionGenerator,'Y = stft(x,"Window",props.Window,"OverlapLength",props.OverlapLength,"FFTLength",props.FFTLength,"FrequencyRange","onesided");')
        addCode(functionGenerator,'Z = reshape(Y,[],numHops*numChannels);')
    end
    if obj.pUsePowerSpectrum
        addCode(functionGenerator,'Zpower = real(Z.*conj(Z));')
        if obj.pUseMagnitudeSpectrum
            addCode(functionGenerator,'Zmagnitude = sqrt(Zpower);')
        end
    elseif obj.pUseMagnitudeSpectrum
        addCode(functionGenerator,'Zmagnitude = abs(Z);')
    end
end
end

function iAddLinearSpectrum(obj,functionGenerator,IsStreaming)
if obj.pCalculateLinearSpectrum
    config = obj.pPipelineParameters;
    addCode(functionGenerator)
    addCode(functionGenerator,'% Linear spectrum')
    if strcmpi(obj.plinearSpectrumType,"Power")
        addCode(functionGenerator,'linearSpectrum = Zpower(config.linearSpectrum.FrequencyBins,:)*config.linearSpectrum.NormalizationFactor;')
    else
        addCode(functionGenerator,'linearSpectrum = Zmagnitude(config.linearSpectrum.FrequencyBins,:)*config.linearSpectrum.NormalizationFactor;')
    end

    adjustFirstBin = config.linearSpectrum.FrequencyBins(1)==1;
    adjustLastBin = (config.linearSpectrum.FrequencyBins(end)==floor(config.FFTLength/2+1)) && (rem(config.FFTLength,2)==0);

    if adjustFirstBin && adjustLastBin
        addCode(functionGenerator,'linearSpectrum([1,end],:) = 0.5*linearSpectrum([1,end],:);')
    elseif ~adjustFirstBin && adjustLastBin
        addCode(functionGenerator,'linearSpectrum(end,:) = 0.5*linearSpectrum(end,:);')
    elseif adjustFirstBin && ~adjustLastBin
        addCode(functionGenerator,'linearSpectrum(1,:) = 0.5*linearSpectrum(1,:);')
    end
    if IsStreaming % STREAMING
        linearSpectrumUsedDownstream = obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,'linearSpectrum');
        if linearSpectrumUsedDownstream && obj.linearSpectrum
            addCode(functionGenerator,'%s = linearSpectrum;',getOutIdx(IsStreaming,'linearSpectrum'))
            addCode(functionGenerator,'linearSpectrum = reshape(linearSpectrum,[],1,numChannels);')
        elseif ~linearSpectrumUsedDownstream && obj.linearSpectrum
            addCode(functionGenerator,'%s = linearSpectrum;',getOutIdx(IsStreaming,'linearSpectrum'))
        elseif linearSpectrumUsedDownstream && ~obj.linearSpectrum
            addCode(functionGenerator,'linearSpectrum = reshape(linearSpectrum,[],1,numChannels);')
        end
    else % BATCH
        addCode(functionGenerator,'linearSpectrum = reshape(linearSpectrum,[],numHops,numChannels);')
        if obj.linearSpectrum
            addCode(functionGenerator,'%s = permute(linearSpectrum,[2,1,3]);',getOutIdx(IsStreaming,'linearSpectrum'))
        end
    end
end
end

function iAddMelSpectrum(obj,functionGenerator,IsStreaming)
if obj.pCalculateMelSpectrum
    addCode(functionGenerator)
    addCode(functionGenerator,'% Mel spectrum')
    audSpec = 'melSpectrum';
    audSpecIsOutput = obj.melSpectrum;
    audSpecIsUsed = obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,'melSpectrum') || obj.pCalculateMFCC;
    audSpecType = obj.pmelSpectrumType;

    addAudSpec(functionGenerator,audSpec,audSpecType,audSpecIsUsed,audSpecIsOutput,IsStreaming)
end
end

function iAddBarkSpectrum(obj,functionGenerator,IsStreaming)
if obj.pCalculateBarkSpectrum
    addCode(functionGenerator)
    addCode(functionGenerator,'% Bark spectrum')
    audSpec = 'barkSpectrum';
    audSpecIsOutput = obj.barkSpectrum;
    audSpecIsUsed = obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,'barkSpectrum');
    audSpecType = obj.pbarkSpectrumType;

    addAudSpec(functionGenerator,audSpec,audSpecType,audSpecIsUsed,audSpecIsOutput,IsStreaming)
end
end

function iAddERBSpectrum(obj,functionGenerator,IsStreaming)
if obj.pCalculateERBSpectrum
    addCode(functionGenerator)
    addCode(functionGenerator,'% ERB spectrum')
    audSpec = 'erbSpectrum';
    audSpecIsOutput = obj.erbSpectrum;
    audSpecIsUsed = obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,'erbSpectrum') || obj.pCalculateGTCC;
    audSpecType = obj.perbSpectrumType;

    addAudSpec(functionGenerator,audSpec,audSpecType,audSpecIsUsed,audSpecIsOutput,IsStreaming)
end
end

function iAddMFCC(obj,functionGenerator,IsStreaming)
if obj.pCalculateMFCC
    config = obj.pPipelineParameters;
    addCode(functionGenerator)
    addCode(functionGenerator,'% Mel-frequency cepstral coefficients (MFCC)')
    if ~obj.mfccDelta && ~obj.mfccDeltaDelta
        addCode(functionGenerator,'%s = cepstralCoefficients(melSpectrum,"NumCoeffs",%d,"Rectification","%s");',getOutIdx(IsStreaming,'mfcc'),config.mfcc.Parameters.NumCoeffs,config.mfcc.Parameters.Rectification)
    else
        addCode(functionGenerator,'melcc = cepstralCoefficients(melSpectrum,"NumCoeffs",%d,"Rectification","%s");',config.mfcc.Parameters.NumCoeffs,config.mfcc.Parameters.Rectification);
        if obj.mfcc
            addCode(functionGenerator,'%s = melcc;',getOutIdx(IsStreaming,'mfcc'))
        end
        if obj.mfccDelta && ~obj.mfccDeltaDelta
            if IsStreaming
                addCode(functionGenerator,'[%s,state.mfccDelta] = audioDelta(melcc,%d,state.mfccDelta);',getOutIdx(IsStreaming,'mfccDelta'),config.mfcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'%s = audioDelta(melcc,%d);',getOutIdx(IsStreaming,'mfccDelta'),config.mfcc.Parameters.DeltaWindowLength)
            end
        elseif obj.mfccDeltaDelta
            if IsStreaming
                addCode(functionGenerator,'[melccDelta,state.mfccDelta] = audioDelta(melcc,%d,state.mfccDelta);',config.mfcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'melccDelta = audioDelta(melcc,%d);',config.mfcc.Parameters.DeltaWindowLength)
            end
            if obj.mfccDelta
                addCode(functionGenerator,'%s = melccDelta;',getOutIdx(IsStreaming,'mfccDelta'))
            end
            if IsStreaming
                addCode(functionGenerator,'[%s,state.mfccDeltaDelta] = audioDelta(melccDelta,%d,state.mfccDeltaDelta);',getOutIdx(IsStreaming,'mfccDeltaDelta'),config.mfcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'%s = audioDelta(melccDelta,%d);',getOutIdx(IsStreaming,'mfccDeltaDelta'),config.mfcc.Parameters.DeltaWindowLength)
            end
        end
    end
end
end

function iAddGTCC(obj,functionGenerator,IsStreaming)
if obj.pCalculateGTCC
    config = obj.pPipelineParameters;
    addCode(functionGenerator)
    addCode(functionGenerator,'% Gammatone-frequency cepstral coefficients (GTCC)')
    if ~obj.gtccDelta && ~obj.gtccDeltaDelta
        addCode(functionGenerator,'%s = cepstralCoefficients(erbSpectrum,"NumCoeffs",%d,"Rectification","%s");',getOutIdx(IsStreaming,'gtcc'),config.gtcc.Parameters.NumCoeffs,config.gtcc.Parameters.Rectification)
    else
        addCode(functionGenerator,'gammacc = cepstralCoefficients(erbSpectrum,"NumCoeffs",%d,"Rectification","%s");',config.gtcc.Parameters.NumCoeffs,config.gtcc.Parameters.Rectification);
        if obj.gtcc
            addCode(functionGenerator,'%s = gammacc;',getOutIdx(IsStreaming,'gtcc'))
        end
        if obj.gtccDelta && ~obj.gtccDeltaDelta
            if IsStreaming
                addCode(functionGenerator,'[%s,state.gtccDelta] = audioDelta(gammacc,%d,state.gtccDelta);',getOutIdx(IsStreaming,'gtccDelta'),config.gtcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'%s = audioDelta(gammacc,%d);',getOutIdx(IsStreaming,'gtccDelta'),config.gtcc.Parameters.DeltaWindowLength)
            end
        elseif obj.gtccDeltaDelta
            if IsStreaming
                addCode(functionGenerator,'[gammaccDelta,state.gtccDelta] = audioDelta(gammacc,%d,state.gtccDelta);',config.gtcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'gammaccDelta = audioDelta(gammacc,%d);',config.gtcc.Parameters.DeltaWindowLength)
            end
            if obj.gtccDelta
                addCode(functionGenerator,'%s = gammaccDelta;',getOutIdx(IsStreaming,'gtccDelta'))
            end
            if IsStreaming
                addCode(functionGenerator,'[%s,state.gtccDeltaDelta] = audioDelta(gammaccDelta,%d,state.gtccDeltaDelta);',getOutIdx(IsStreaming,'gtccDeltaDelta'),config.gtcc.Parameters.DeltaWindowLength)
            else
                addCode(functionGenerator,'%s = audioDelta(gammaccDelta,%d);',getOutIdx(IsStreaming,'gtccDeltaDelta'),config.gtcc.Parameters.DeltaWindowLength)
            end
        end
    end
end
end

function iAddSpectralMoments(obj,functionGenerator,IsStreaming)
switch   num2str([obj.spectralCentroid, obj.spectralSpread, obj.spectralSkewness, obj.spectralKurtosis])
    case num2str([true false false false])
        addCode(functionGenerator,'%s = spectralCentroid(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([false true false false])
        addCode(functionGenerator,'%s = spectralSpread(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            obj.SpectralDescriptorInput)

    case num2str([false false true false])
        addCode(functionGenerator,'%s = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            obj.SpectralDescriptorInput)

    case num2str([false false false true])
        addCode(functionGenerator,'%s = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            obj.SpectralDescriptorInput)

    case num2str([true true false false])
        addCode(functionGenerator,'[%s,%s] = spectralSpread(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([true false true false])
        addCode(functionGenerator,'[%s,~,%s] = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([true false false true])
        addCode(functionGenerator,'[%s,~,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([false true true false])
        addCode(functionGenerator,'[%s,%s] = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            obj.SpectralDescriptorInput)

    case num2str([false true false true])
        addCode(functionGenerator,'[%s,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            obj.SpectralDescriptorInput)

    case num2str([false false true true])
        addCode(functionGenerator,'%s = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            obj.SpectralDescriptorInput)
        addCode(functionGenerator,'%s = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            obj.SpectralDescriptorInput)

    case num2str([false true true true])
        addCode(functionGenerator,'[%s,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            obj.SpectralDescriptorInput)
        addCode(functionGenerator,'%s = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            obj.SpectralDescriptorInput)

    case num2str([true false true true])
        addCode(functionGenerator,'[%s,~,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)
        addCode(functionGenerator,'%s = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            obj.SpectralDescriptorInput)

    case num2str([true true false true])
        addCode(functionGenerator,'[%s,%s,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([true true true false])
        addCode(functionGenerator,'[%s,%s,%s] = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)

    case num2str([true true true true])
        addCode(functionGenerator,'[%s,%s,%s] = spectralKurtosis(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralKurtosis'), ...
            getOutIdx(IsStreaming,'spectralSpread'), ...
            getOutIdx(IsStreaming,'spectralCentroid'), ...
            obj.SpectralDescriptorInput)
        addCode(functionGenerator,'%s = spectralSkewness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
            getOutIdx(IsStreaming,'spectralSkewness'), ...
            obj.SpectralDescriptorInput)

    otherwise %[false false false false]
        % do nothing
end
end

function iAddSpectralDescriptor(obj,functionGenerator,IsStreaming,descriptor)
config = obj.pPipelineParameters;
if obj.(descriptor)
    switch descriptor
        case 'spectralCrest'
            addCode(functionGenerator,'%s = spectralCrest(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
                getOutIdx(IsStreaming,'spectralCrest'),obj.SpectralDescriptorInput)
        case 'spectralDecrease'
            addCode(functionGenerator,'%s = spectralDecrease(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
                getOutIdx(IsStreaming,'spectralDecrease'),obj.SpectralDescriptorInput)
        case 'spectralEntropy'
            addCode(functionGenerator,'%s = spectralEntropy(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
                getOutIdx(IsStreaming,'spectralEntropy'),obj.SpectralDescriptorInput)
        case 'spectralFlatness'
            addCode(functionGenerator,'%s = spectralFlatness(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
                getOutIdx(IsStreaming,'spectralFlatness'),obj.SpectralDescriptorInput)
        case 'spectralFlux'
            if IsStreaming
                addCode(functionGenerator,'[%s, state.spectralFlux] = spectralFlux(%s,config.SpectralDescriptorInput.FrequencyVector,state.spectralFlux,"NormType",%s);', ...
                    getOutIdx(IsStreaming,'spectralFlux'),obj.SpectralDescriptorInput,mat2str(config.spectralFlux.Parameters.NormType))
            else
                addCode(functionGenerator,'%s = spectralFlux(%s,config.SpectralDescriptorInput.FrequencyVector,"NormType",%d);', ...
                    getOutIdx(IsStreaming,'spectralFlux'),obj.SpectralDescriptorInput,config.spectralFlux.Parameters.NormType)
            end
        case 'spectralRolloffPoint'
            addCode(functionGenerator,'%s = spectralRolloffPoint(%s,config.SpectralDescriptorInput.FrequencyVector,"Threshold",%d);', ...
                getOutIdx(IsStreaming,'spectralRolloffPoint'),obj.SpectralDescriptorInput,config.spectralRolloffPoint.Parameters.Threshold)
        case 'spectralSlope'
            addCode(functionGenerator,'%s = spectralSlope(%s,config.SpectralDescriptorInput.FrequencyVector);', ...
                getOutIdx(IsStreaming,'spectralSlope'),obj.SpectralDescriptorInput)
    end
end
end

function iAddPeriodicityMetrics(obj,functionGenerator,IsStreaming,UsesState,UsesWindow,UsesBuffer,PreBufferPitch)
config = obj.pPipelineParameters;
if obj.pitch || obj.harmonicRatio || obj.zerocrossrate
    addCode(functionGenerator)
    addCode(functionGenerator,'% Periodicity features')
    if obj.pitch
        if PreBufferPitch
            addCode(functionGenerator,'f0 = pitch(xb2,props.SampleRate,"WindowLength",%s,"OverlapLength",0,"Method",''%s'',"Range",coder.const(cast(%s,"like",x)));',iCreateWindowSpecifier(UsesWindow),config.pitch.Parameters.Method,mat2str(config.pitch.Parameters.Range))
            addCode(functionGenerator,'%s = reshape(f0,numHops,1,numChannels);',getOutIdx(IsStreaming,'pitch'))
        else
            addCode(functionGenerator,'%s = pitch(x,props.SampleRate,"WindowLength",%s,"OverlapLength",props.OverlapLength,"Method",''%s'',"Range",coder.const(cast(%s,"like",x)));',getOutIdx(IsStreaming,'pitch'),iCreateWindowSpecifier(UsesWindow),config.pitch.Parameters.Method,mat2str(config.pitch.Parameters.Range))
        end
    end
    if obj.harmonicRatio
        if UsesBuffer
            addCode(functionGenerator,'hr = harmonicRatio(xb2w,props.SampleRate,"Window",ones(numel(props.Window),1,"like",x),"OverlapLength",0);',getOutIdx(IsStreaming,'harmonicRatio'))
            addCode(functionGenerator,'%s = reshape(hr,numHops,1,numChannels);',getOutIdx(IsStreaming,'harmonicRatio'))
        else
            if IsStreaming
                addCode(functionGenerator,'%s = harmonicRatio(x,props.SampleRate,"Window",props.Window,"OverlapLength",0);',getOutIdx(IsStreaming,'harmonicRatio'))
            else
                addCode(functionGenerator,'%s = harmonicRatio(x,props.SampleRate,"Window",props.Window,"OverlapLength",props.OverlapLength);',getOutIdx(IsStreaming,'harmonicRatio'))
            end
        end
    end
    if obj.zerocrossrate
        if IsStreaming && UsesState
            addCode(functionGenerator,'%s = zerocrossrate(x,"InitialState",state.zerocrossrate,"WindowLength",%s,"OverlapLength",0,"ZeroPositive",%s,"Level",%s,"Method",''%s'',"Threshold",%s,"TransitionEdge",''%s'');', ...
                getOutIdx(IsStreaming,'zerocrossrate'),iCreateWindowSpecifier(UsesWindow),num2str(config.zerocrossrate.Parameters.ZeroPositive),num2str(config.zerocrossrate.Parameters.Level),config.zerocrossrate.Parameters.Method,num2str(config.zerocrossrate.Parameters.Threshold),config.zerocrossrate.Parameters.TransitionEdge);
            addCode(functionGenerator,'state.zerocrossrate = x(%s - props.OverlapLength,:);',iCreateWindowSpecifier(UsesWindow))
        else
            if UsesBuffer
                addCode(functionGenerator,'zcr = zerocrossrate(xb2,"InitialState",reshape([zeros(1,1,numChannels),xb3(%s - props.OverlapLength,1:end-1,:)],1,[]),"OverlapLength",0,"ZeroPositive",%s,"Level",%s,"Method",''%s'',"Threshold",%s,"TransitionEdge",''%s'');', ...
                    iCreateWindowSpecifier(UsesWindow),num2str(config.zerocrossrate.Parameters.ZeroPositive),num2str(config.zerocrossrate.Parameters.Level),config.zerocrossrate.Parameters.Method,num2str(config.zerocrossrate.Parameters.Threshold),config.zerocrossrate.Parameters.TransitionEdge);
                addCode(functionGenerator,'%s = reshape(zcr,numHops,1,[]);',getOutIdx(IsStreaming,'zerocrossrate'))
            else
                addCode(functionGenerator,'%s = zerocrossrate(x,"WindowLength",%s,"OverlapLength",props.OverlapLength,"ZeroPositive",%s,"Level",%s,"Method",''%s'',"Threshold",%s,"TransitionEdge",''%s'');', ...
                    getOutIdx(IsStreaming,'zerocrossrate'),iCreateWindowSpecifier(UsesWindow),num2str(config.zerocrossrate.Parameters.ZeroPositive),num2str(config.zerocrossrate.Parameters.Level),config.zerocrossrate.Parameters.Method,num2str(config.zerocrossrate.Parameters.Threshold),config.zerocrossrate.Parameters.TransitionEdge);
            end
        end
    end
end
end
function iAddEnergyMetrics(obj,functionGenerator,IsStreaming)
config = obj.pPipelineParameters;
if obj.shortTimeEnergy
    addCode(functionGenerator)
    addCode(functionGenerator,'% Energy features')
    if obj.pUseSpectrum
        evenLengthFFT = ~rem(config.FFTLength,2);
        if evenLengthFFT
            astring = sprintf('%s = reshape((2*sum(Zpower,1) - Zpower(1,:,:) - Zpower(end,:,:))./props.FFTLength,1,[],numChannels);',getOutIdx(IsStreaming,'shortTimeEnergy'));
        else
            astring = sprintf('%s = reshape((2*sum(Zpower,1) - Zpower(1,:,:))./props.FFTLength,1,[],numChannels);',getOutIdx(IsStreaming,'shortTimeEnergy'));
        end
    else
        if IsStreaming
            astring = sprintf('%s = sum(bsxfun(@times,x,props.Window).^2,1);',getOutIdx(IsStreaming,'shortTimeEnergy'));
        else
            astring = sprintf('%s = sum(xb3w.^2,1);',getOutIdx(IsStreaming,'shortTimeEnergy'));
        end
    end
    addCode(functionGenerator,astring)
end
end
function getPropsFunctionGenerator = iAddSubfunction_getProps(obj,functionGenerator,IsStreaming,UsesWindow,precision)
getPropsFunctionGenerator = sigutils.internal.emission.MatlabFunctionGenerator;
getPropsFunctionGenerator.AlignEndOfLineComments = false;
getPropsFunctionGenerator.Name = 'getProps';
getPropsFunctionGenerator.InputArgs = {'dataType'};
getPropsFunctionGenerator.OutputArgs = {'props'};
getPropsFunctionGenerator.CoderCompatible = true;
getPropsFunctionGenerator.RCSRevisionAndDate = false;
getPropsFunctionGenerator.EndOfFileMarker = false;
if UsesWindow
    addCode(getPropsFunctionGenerator,'props.Window = cast(%s,dataType);',mat2str(obj.Window(:),precision));
else % Only uses window length, no need to print entire window.
    addCode(getPropsFunctionGenerator,'props.WindowLength = cast(%d,dataType);',numel(obj.Window));
end
if ~IsStreaming || obj.pitch || obj.zerocrossrate
    addCode(getPropsFunctionGenerator,'props.OverlapLength = cast(%d,dataType);',obj.OverlapLength);
end
addCode(getPropsFunctionGenerator,'props.SampleRate = cast(%d,dataType);',obj.SampleRate);
if obj.pUseSpectrum
    if obj.shortTimeEnergy
        addCode(getPropsFunctionGenerator,'props.FFTLength = cast(%d,dataType);',obj.pFFTLength);
    else
        addCode(getPropsFunctionGenerator,'props.FFTLength = %s(%d);',getIndexingClass(obj.pFFTLength),obj.pFFTLength);
    end
end
addCode(getPropsFunctionGenerator,'props.NumFeatures = %s(%d);',getIndexingClass(obj.pPipelineParameters.FeaturesPerHop),obj.pPipelineParameters.FeaturesPerHop);
addLocalFunction(functionGenerator,getPropsFunctionGenerator);
end

function getStateFunctionGenerator = iAddSubfunction_getState(obj,functionGenerator,IsStreaming,UsesState)
if IsStreaming && UsesState
    config = obj.pPipelineParameters;
    getStateFunctionGenerator = sigutils.internal.emission.MatlabFunctionGenerator;
    getStateFunctionGenerator.AlignEndOfLineComments = false;
    getStateFunctionGenerator.Name = 'getState';
    getStateFunctionGenerator.InputArgs = {'dataType','numChannels'};
    getStateFunctionGenerator.OutputArgs = {'state'};
    getStateFunctionGenerator.CoderCompatible = true;
    getStateFunctionGenerator.RCSRevisionAndDate = false;
    getStateFunctionGenerator.EndOfFileMarker = false;
    addCode(getStateFunctionGenerator,'state.NumChannels = numChannels;');
    if obj.spectralFlux
        addCode(getStateFunctionGenerator,'state.spectralFlux = zeros(%d,numChannels,dataType);',numel(config.SpectralDescriptorInput.FrequencyVector));
    end
    if obj.mfccDelta || obj.mfccDeltaDelta
        addCode(getStateFunctionGenerator,'state.mfccDelta = zeros(%d,%d,numChannels,dataType);',config.mfcc.Parameters.DeltaWindowLength-1,config.mfcc.Parameters.NumCoeffs);
    end
    if obj.mfccDeltaDelta
        addCode(getStateFunctionGenerator,'state.mfccDeltaDelta = zeros(%d,%d,numChannels,dataType);',config.mfcc.Parameters.DeltaWindowLength-1,config.mfcc.Parameters.NumCoeffs);
    end
    if obj.gtccDelta || obj.gtccDeltaDelta
        addCode(getStateFunctionGenerator,'state.gtccDelta = zeros(%d,%d,numChannels,dataType);',config.gtcc.Parameters.DeltaWindowLength-1,config.gtcc.Parameters.NumCoeffs);
    end
    if obj.gtccDeltaDelta
        addCode(getStateFunctionGenerator,'state.gtccDeltaDelta = zeros(%d,%d,numChannels,dataType);',config.gtcc.Parameters.DeltaWindowLength-1,config.gtcc.Parameters.NumCoeffs);
    end
    if obj.zerocrossrate
        addCode(getStateFunctionGenerator,'state.zerocrossrate = zeros(1,numChannels,dataType);');
    end
    addLocalFunction(functionGenerator,getStateFunctionGenerator)
end
end
function resetStateFunctionGenerator = iAddSubfunction_reset(obj,functionGenerator,IsStreaming,UsesState)
if IsStreaming && UsesState
    resetStateFunctionGenerator = sigutils.internal.emission.MatlabFunctionGenerator;
    resetStateFunctionGenerator.Name = 'reset';
    resetStateFunctionGenerator.InputArgs = {'state'};
    resetStateFunctionGenerator.OutputArgs = {'state'};
    resetStateFunctionGenerator.CoderCompatible = true;
    resetStateFunctionGenerator.RCSRevisionAndDate = false;
    resetStateFunctionGenerator.EndOfFileMarker = false;
    if obj.spectralFlux
        addCode(resetStateFunctionGenerator,'state.spectralFlux(:,:) = 0;')
    end
    if obj.mfccDelta || obj.mfccDeltaDelta
        addCode(resetStateFunctionGenerator,'state.mfccDelta(:,:,:) = 0;')
    end
    if obj.mfccDeltaDelta
        addCode(resetStateFunctionGenerator,'state.mfccDeltaDelta(:,:,:) = 0;')
    end
    if obj.gtccDelta || obj.gtccDeltaDelta
        addCode(resetStateFunctionGenerator,'state.gtccDelta(:,:,:) = 0;')
    end
    if obj.gtccDeltaDelta
        addCode(resetStateFunctionGenerator,'state.gtccDeltaDelta(:,:,:) = 0;')
    end
    if obj.zerocrossrate
        addCode(resetStateFunctionGenerator,'state.zerocrossrate(:,:) = 0;')
    end
    addLocalFunction(functionGenerator,resetStateFunctionGenerator)
end
end
function iAddSubfunction_getConfig(obj,functionGenerator,IsStreaming,UsesConfig,precision)
config = obj.pPipelineParameters;
getConfigFunctionGenerator = sigutils.internal.emission.MatlabFunctionGenerator;
getConfigFunctionGenerator.AlignEndOfLineComments = false;
getConfigFunctionGenerator.Name = 'getConfig';
if ~UsesConfig
    getConfigFunctionGenerator.InputArgs = {};
    getConfigFunctionGenerator.OutputArgs = {'outputIndex'};
elseif obj.linearSpectrum && numel(fields(obj.pOutputIndexReduced))==1 && ~obj.pUseWindowNormalizationPower && ~obj.pUseWindowNormalizationMagnitude
    getConfigFunctionGenerator.InputArgs = {'dataType'};
    getConfigFunctionGenerator.OutputArgs = {'config','outputIndex'};
else
    getConfigFunctionGenerator.InputArgs = {'dataType','props'};
    getConfigFunctionGenerator.OutputArgs = {'config','outputIndex'};
end
getConfigFunctionGenerator.CoderCompatible = true;
getConfigFunctionGenerator.RCSRevisionAndDate = false;
getConfigFunctionGenerator.EndOfFileMarker = false;
if obj.pUseWindowNormalizationPower
    addCode(getConfigFunctionGenerator,'powerNormalizationFactor = 1/(sum(props.Window)^2);');
end
if obj.pUseWindowNormalizationMagnitude
    addCode(getConfigFunctionGenerator,'magnitudeNormalizationFactor = 1/sum(props.Window);');
end

if obj.pUseSpectrum && IsStreaming
    addCode(getConfigFunctionGenerator)
    if isscalar(obj.pPipelineParameters.OneSidedSpectrumBins)
        addCode(getConfigFunctionGenerator,'config.OneSidedSpectrumBins = %s(%d);',getIndexingClass(max(obj.pPipelineParameters.OneSidedSpectrumBins)),obj.pPipelineParameters.OneSidedSpectrumBins);
    else
        addCode(getConfigFunctionGenerator,'config.OneSidedSpectrumBins = %s(%d:%d);',getIndexingClass(max(obj.pPipelineParameters.OneSidedSpectrumBins)),obj.pPipelineParameters.OneSidedSpectrumBins(1),obj.pPipelineParameters.OneSidedSpectrumBins(end));
    end
end

% Linear Spectrum Parameters
iAddLinearSpectrum_getConfig(obj,getConfigFunctionGenerator)

% Mel Spectrum Parameters
iAddMelSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)

% Bark Spectrum Parameters
iAddBarkSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)

% ERB Spectrum Parameters
iAddERBSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)

% Spectral Descriptor Parameters
if obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,"linearSpectrum")
    params  = getFeatureExtractorParams(obj,"linearSpectrum");
    range   = params.FrequencyRange;
    binHigh = floor(range(2)*obj.pFFTLength/obj.SampleRate + 1);
    addCode(getConfigFunctionGenerator)
    addCode(getConfigFunctionGenerator,'FFTLength = cast(props.FFTLength,''like'',props.SampleRate);')
    addCode(getConfigFunctionGenerator,'w = (props.SampleRate/FFTLength)*(linearSpectrumFrequencyBins-1);');
    if rem(obj.pFFTLength,2) && binHigh == floor(obj.pFFTLength/2 + 1)
        addCode(getConfigFunctionGenerator,'w(end) = props.SampleRate*(FFTLength-1)/(2*FFTLength);');
    end
    addCode(getConfigFunctionGenerator,'config.SpectralDescriptorInput.FrequencyVector = cast(w(:),dataType);')
end

% Determine data type of indexing
idxDataType = getIndexingClass(config.FeaturesPerHop);
% Add Indexing
outputIndex = obj.pOutputIndexReduced;
outputIdxFields = fields(outputIndex);
if obj.pUseSpectrum
    addCode(getConfigFunctionGenerator)
end
for ii = 1:numel(outputIdxFields)
    val = outputIndex.(outputIdxFields{ii});
    if isscalar(val)
        addCode(getConfigFunctionGenerator,'outputIndex.%s = %s(%d);',outputIdxFields{ii},idxDataType,val)
    else
        addCode(getConfigFunctionGenerator,'outputIndex.%s = %s(%d:%d);',outputIdxFields{ii},idxDataType,val(1),val(end))
    end
end

addLocalFunction(functionGenerator,getConfigFunctionGenerator);
end
function iAddLinearSpectrum_getConfig(obj,getConfigFunctionGenerator)
config = obj.pPipelineParameters;
if obj.pCalculateLinearSpectrum
    addCode(getConfigFunctionGenerator)
    if obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,"linearSpectrum")
        addCode(getConfigFunctionGenerator,'linearSpectrumFrequencyBins = %d:%d;',config.linearSpectrum.FrequencyBins(1),config.linearSpectrum.FrequencyBins(end));
        addCode(getConfigFunctionGenerator,'config.linearSpectrum.FrequencyBins = %s(linearSpectrumFrequencyBins);',getIndexingClass(max(config.linearSpectrum.FrequencyBins)));
    else
        if isscalar(config.linearSpectrum.FrequencyBins)
            addCode(getConfigFunctionGenerator,'config.linearSpectrum.FrequencyBins = %s(%d);',getIndexingClass(config.linearSpectrum.FrequencyBins),config.linearSpectrum.FrequencyBins);
        else
            addCode(getConfigFunctionGenerator,'config.linearSpectrum.FrequencyBins = %s(%d:%d);',getIndexingClass(config.linearSpectrum.FrequencyBins(end)),config.linearSpectrum.FrequencyBins(1),config.linearSpectrum.FrequencyBins(end));
        end
    end
    if obj.plinearSpectrumWinNorm
        if strcmpi(obj.plinearSpectrumType,"Magnitude")
            addCode(getConfigFunctionGenerator,'config.linearSpectrum.NormalizationFactor = cast(2*magnitudeNormalizationFactor,dataType);');
        else
            addCode(getConfigFunctionGenerator,'config.linearSpectrum.NormalizationFactor = cast(2*powerNormalizationFactor,dataType);');
        end
    else
        addCode(getConfigFunctionGenerator,'config.linearSpectrum.NormalizationFactor = cast(2,dataType);');
    end
end
end
function iAddMelSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)
if obj.pCalculateMelSpectrum
    addCode(getConfigFunctionGenerator)
    params = getFeatureExtractorParams(obj,"melSpectrum");
    if obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,"melSpectrum")
        addCode(getConfigFunctionGenerator,'[melFilterbank,config.SpectralDescriptorInput.FrequencyVector] = designAuditoryFilterBank(props.SampleRate, ...')
    else
        addCode(getConfigFunctionGenerator,'melFilterbank = designAuditoryFilterBank(props.SampleRate, ...')
    end
    addCode(getConfigFunctionGenerator,'"FrequencyScale","mel", ...')
    addCode(getConfigFunctionGenerator,'"FFTLength",props.FFTLength, ...')
    addCode(getConfigFunctionGenerator,'"OneSided",true, ...')
    addCode(getConfigFunctionGenerator,'"FrequencyRange",%s, ...',mat2str(params.FrequencyRange,precision))
    addCode(getConfigFunctionGenerator,'"NumBands",%d, ...',params.NumBands)
    addCode(getConfigFunctionGenerator,'"Normalization","%s", ...',params.FilterBankNormalization)
    addCode(getConfigFunctionGenerator,'"FilterBankDesignDomain","%s");',params.FilterBankDesignDomain)
    if obj.pmelSpectrumWinNorm
        if strcmpi(obj.pmelSpectrumType,"Power")
            specType = "powerNormalizationFactor";
        else
            specType = "magnitudeNormalizationFactor";
        end
        addCode(getConfigFunctionGenerator,'melFilterbank = melFilterbank*%s;',specType)
    end
    addCode(getConfigFunctionGenerator,'config.melSpectrum.FilterBank = cast(melFilterbank,dataType);')
end
end
function iAddBarkSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)
if obj.pCalculateBarkSpectrum
    addCode(getConfigFunctionGenerator)
    params = getFeatureExtractorParams(obj,"barkSpectrum");
    if obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,"barkSpectrum")
        addCode(getConfigFunctionGenerator,'[barkFilterbank,config.SpectralDescriptorInput.FrequencyVector] = designAuditoryFilterBank(props.SampleRate, ...')
    else
        addCode(getConfigFunctionGenerator,'barkFilterbank = designAuditoryFilterBank(props.SampleRate, ...')
    end
    addCode(getConfigFunctionGenerator,'"FrequencyScale","bark", ...')
    addCode(getConfigFunctionGenerator,'"FFTLength",props.FFTLength, ...')
    addCode(getConfigFunctionGenerator,'"OneSided",true, ...')
    addCode(getConfigFunctionGenerator,'"FrequencyRange",%s, ...',mat2str(params.FrequencyRange,precision))
    addCode(getConfigFunctionGenerator,'"NumBands",%d, ...',params.NumBands)
    addCode(getConfigFunctionGenerator,'"Normalization","%s", ...',params.FilterBankNormalization)
    addCode(getConfigFunctionGenerator,'"FilterBankDesignDomain","%s");',params.FilterBankDesignDomain)

    if obj.pbarkSpectrumWinNorm
        if strcmpi(obj.pbarkSpectrumType,"Power")
            specType = "powerNormalizationFactor";
        else
            specType = "magnitudeNormalizationFactor";
        end
        addCode(getConfigFunctionGenerator,'barkFilterbank = barkFilterbank*%s;',specType)
    end
    addCode(getConfigFunctionGenerator,'config.barkSpectrum.FilterBank = cast(barkFilterbank,dataType);')
end
end
function iAddERBSpectrum_getConfig(obj,getConfigFunctionGenerator,precision)
if obj.pCalculateERBSpectrum
    addCode(getConfigFunctionGenerator)
    params = getFeatureExtractorParams(obj,"erbSpectrum");
    if obj.pExtractSpectralDescriptor && strcmpi(obj.SpectralDescriptorInput,"erbSpectrum")
        addCode(getConfigFunctionGenerator,'[erbFilterbank,config.SpectralDescriptorInput.FrequencyVector] = coder.const(@feval,''designAuditoryFilterBank'',props.SampleRate, ...')
    else
        addCode(getConfigFunctionGenerator,'erbFilterbank = coder.const(@feval,''designAuditoryFilterBank'',props.SampleRate, ...')
    end
    addCode(getConfigFunctionGenerator,'"FrequencyScale","erb", ...')
    addCode(getConfigFunctionGenerator,'"FFTLength",props.FFTLength, ...')
    addCode(getConfigFunctionGenerator,'"OneSided",true, ...')
    addCode(getConfigFunctionGenerator,'"FrequencyRange",%s, ...',mat2str(params.FrequencyRange,precision))
    addCode(getConfigFunctionGenerator,'"NumBands",%d, ...',params.NumBands)
    addCode(getConfigFunctionGenerator,'"Normalization","%s");',params.FilterBankNormalization)
    if obj.perbSpectrumWinNorm
        if strcmpi(obj.perbSpectrumType,"Power")
            specType = "powerNormalizationFactor";
        else
            specType = "magnitudeNormalizationFactor";
        end
        addCode(getConfigFunctionGenerator,'erbFilterbank = erbFilterbank*%s;',specType)
    end
    addCode(getConfigFunctionGenerator,'config.erbSpectrum.FilterBank = cast(erbFilterbank,dataType);')
end
end
function windowSpecifier = iCreateWindowSpecifier(UsesWindow)
if UsesWindow
    windowSpecifier = 'numel(props.Window)';
else
    windowSpecifier = 'props.WindowLength';
end
end
function tf = iCheckUsesState(obj)
tf = obj.spectralFlux || obj.mfccDelta || obj.mfccDeltaDelta || ...
    obj.gtccDelta || obj.gtccDeltaDelta || obj.zerocrossrate;
end
function tf = iCheckUsesConfig(obj)
tf = obj.linearSpectrum || obj.melSpectrum || ...
    obj.barkSpectrum || obj.erbSpectrum || ...
    obj.mfcc || obj.mfccDelta || obj.mfccDeltaDelta || ...
    obj.gtcc || obj.gtccDelta || obj.gtccDeltaDelta || ...
    obj.spectralCentroid || obj.spectralCrest || ...
    obj.spectralDecrease || obj.spectralEntropy || ...
    obj.spectralFlatness || obj.spectralFlux || ...
    obj.spectralKurtosis || obj.spectralRolloffPoint ...
    || obj.spectralSkewness || obj.spectralSlope || obj.spectralSpread;
end
function tf = iCheckUsesWindow(obj)
tf = iCheckUsesConfig(obj) || obj.harmonicRatio || obj.shortTimeEnergy;
end
function [UseBuffer,PreBufferPitch,UseBuffer3DWindow,UseBuffer3DNoWindow,UseBuffer2DWindow] = iCheckBufferUsage(obj,IsStreaming)
pitchParams = getFeatureExtractorParams(obj,'pitch');
PreBufferPitch = obj.pitch && pitchParams.MedianFilterLength == 1 && ~strcmpi(pitchParams.Method,'SRH') && ~IsStreaming;
UseBuffer = ( (obj.shortTimeEnergy && ~obj.pUseSpectrum) ...
    || ((obj.zerocrossrate + obj.harmonicRatio + PreBufferPitch) >= 2) ) ...
    && ~IsStreaming; % The baseline buffer is 2D no window.
PreBufferPitch = PreBufferPitch && UseBuffer;
UseBuffer3DWindow = UseBuffer && obj.shortTimeEnergy && ~obj.pUseSpectrum;
UseBuffer3DNoWindow = UseBuffer && obj.zerocrossrate;
UseBuffer2DWindow = UseBuffer && obj.harmonicRatio;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%            Do NOT autoindent below         %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function iCreateFunctionHelp(obj,functionGenerator,IsStreaming,UsesState,functionName)
if ~IsStreaming
    functionGenerator.H1Line = 'Extract multiple features from batch audio';
    functionGenerator.Help = { ...
sprintf('featureVector = %s(audioIn) returns audio features',functionName)                         , ...
        'extracted from audioIn.'                                                                  , ...
        ' '                                                                                        , ...
        'Parameters of the audioFeatureExtractor used to generate this '                           , ...
        'function must be honored when calling this function.'                                     , ...
sprintf(' - Sample rate of the input should be %d Hz.',obj.SampleRate)                             , ...
sprintf(' - Input frame length should be greater than or equal to %d samples.',numel(obj.Window))  , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 1: Extract features'                                                         , ...
sprintf('     source = dsp.ColoredNoise("SamplesPerFrame",%d);',obj.SampleRate)                    , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
sprintf('         featureArray = %s(audioIn);',functionName)                                       , ...
        '         % ... do something with featureArray ...'                                        , ...
        '     end'                                                                                 , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 2: Generate code'                                                            , ...
        '     targetDataType = "single";'                                                          , ...
sprintf('     codegen %s -args {ones(%d,1,targetDataType)}',functionName,obj.SampleRate)           , ...
sprintf('     source = dsp.ColoredNoise("SamplesPerFrame",%d, ...',obj.SampleRate)                 , ...
        '                               "OutputDataType",targetDataType);'                         , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
sprintf('         featureArray = %s_mex(audioIn);',functionName)                                   , ...
        '         % ... do something with featureArray ...'                                        , ...
        '     end'                                                                                 , ...
        };
elseif IsStreaming && UsesState
    functionGenerator.InputArgs = {'x','varargin'};
    functionGenerator.H1Line = 'Extract multiple features from streaming audio';
    functionGenerator.Help = { ...
sprintf('featureVector = %s(audioIn) returns audio features',functionName)                         , ...
        'extracted from audioIn.'                                                                  , ...
        ' '                                                                                        , ...
sprintf('featureVector = %s(audioIn,"Reset",TF) returns feature extractors',functionName)          , ...
        'to their initial conditions before extracting features.'                                  , ...
        ' '                                                                                        , ...
        'Parameters of the audioFeatureExtractor used to generate this '                           , ...
        'function must be honored when calling this function. '                                    , ...
sprintf(' - Sample rate of the input should be %d Hz.',obj.SampleRate)                             , ...
sprintf(' - Frame length of the input should be %d samples.',numel(obj.Window))                    , ...
        ' - Successive frames of the input should be overlapped by'                                , ...
sprintf('   %d samples before calling %s.',obj.OverlapLength,functionName)                         , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 1: Extract features'                                                         , ...
        '     source = dsp.ColoredNoise();'                                                        , ...
        '     inputBuffer = dsp.AsyncBuffer;'                                                      , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
        '         write(inputBuffer,audioIn);'                                                     , ...
sprintf('         while inputBuffer.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)    , ...
sprintf('             x = read(inputBuffer,%d,%d);',numel(obj.Window),obj.OverlapLength)           , ...
sprintf('             featureVector = %s(x);',functionName)                                        , ...
        '             % ... do something with featureVector ...'                                   , ...
        '         end'                                                                             , ...
        '      end'                                                                                , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 2: Extract features from speech regions only'                                , ...
        '     [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");'                       , ...
sprintf('     audioIn = resample(audioIn,%d,fs);',obj.SampleRate)                                  , ...
        '     source = dsp.AsyncBuffer(size(audioIn,1));'                                          , ...
        '     write(source,audioIn);'                                                              , ...
        '     TF = false;'                                                                         , ...
sprintf('     while source.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)             , ...
sprintf('         x = read(source,%d,%d);',numel(obj.Window),obj.OverlapLength)                    , ...
        '         isSilence = var(x) < 0.01;'                                                      , ...
        '         if ~isSilence'                                                                   , ...
sprintf('             featureVector = %s(x,"Reset",TF);',functionName)                             , ...
        '             TF = false;'                                                                 , ...
        '         else'                                                                            , ...
        '             TF = true;'                                                                  , ...
        '         end'                                                                             , ...
        '         % ... do something with featureVector ...'                                       , ...
        '     end'                                                                                 , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 3: Generate code that does not use reset'                                    , ...
        '     targetDataType = "single";'                                                          , ...
sprintf('     codegen %s -args {ones(%d,1,targetDataType)}',functionName,numel(obj.Window))        , ...
        '     source = dsp.ColoredNoise(''OutputDataType'',targetDataType);'                       , ...
        '     inputBuffer = dsp.AsyncBuffer;'                                                      , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
        '         write(inputBuffer,audioIn);'                                                     , ...
sprintf('         while inputBuffer.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)    , ...
sprintf('             x = read(inputBuffer,%d,%d);',numel(obj.Window),obj.OverlapLength)           , ...
sprintf('             featureVector = %s_mex(x);',functionName)                                    , ...
        '             % ... do something with featureVector ...'                                   , ...
        '         end'                                                                             , ...
        '      end'                                                                                , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 4: Generate code that uses reset'                                            , ...
        '     targetDataType = "single";'                                                          , ...
sprintf('     codegen %s -args {ones(%d,1,targetDataType),''Reset'',true}',functionName,numel(obj.Window)), ...
        '     [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");'                       , ...
sprintf('     audioIn = resample(audioIn,%d,fs);',obj.SampleRate)                                  , ...
        '     source = dsp.AsyncBuffer(size(audioIn,1));'                                          , ...
        '     write(source,cast(audioIn,targetDataType));'                                         , ...
        '     TF = false;'                                                                         , ...
sprintf('     while source.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)             , ...
sprintf('         x = read(source,%d,%d);',numel(obj.Window),obj.OverlapLength)                    , ...
        '         isSilence = var(x) < 0.01;'                                                      , ...
        '         if ~isSilence'                                                                   , ...
sprintf('             featureVector = %s_mex(x,''Reset'',TF);',functionName)                       , ...
        '             TF = false;'                                                                 , ...
        '         else'                                                                            , ...
        '             TF = true;'                                                                  , ...
        '         end'                                                                             , ...
        '         % ... do something with featureVector ...'                                       , ...
        '     end'                                                                                 , ...
        };
else %IsStreaming && ~UsesState
    functionGenerator.H1Line = 'Extract multiple features from streaming audio';
    functionGenerator.Help = { ...
sprintf('featureVector = %s(audioIn) returns audio features',functionName)                         , ...
        'extracted from audioIn.'                                                                  , ...
        ' '                                                                                        , ...
        'Parameters of the audioFeatureExtractor used to generated this '                          , ...
        'function must be honored when calling this function. '                                    , ...
sprintf(' - Sample rate of the input should be %d Hz.',obj.SampleRate)                             , ...
sprintf(' - Frame length of the input should be %d samples.',numel(obj.Window))                    , ...
        ' - Successive frames of the input should be overlapped by'                                , ...
sprintf('   %d samples before calling %s.',obj.OverlapLength,functionName)                         , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 1: Extract features'                                                         , ...
        '     source = dsp.ColoredNoise();'                                                        , ...
        '     inputBuffer = dsp.AsyncBuffer;'                                                      , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
        '         write(inputBuffer,audioIn);'                                                     , ...
sprintf('         while inputBuffer.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)    , ...
sprintf('             x = read(inputBuffer,%d,%d);',numel(obj.Window),obj.OverlapLength)           , ...
sprintf('             featureVector = %s(x);',functionName)                                        , ...
        '             % ... do something with featureVector ...'                                   , ...
        '         end'                                                                             , ...
        '      end'                                                                                , ...
        ' '                                                                                        , ...
        ' '                                                                                        , ...
        '   % EXAMPLE 2: Generate code'                                                            , ...
        '     targetDataType = "single";'                                                          , ...
sprintf('     codegen %s -args {ones(%d,1,targetDataType)}',functionName,numel(obj.Window))        , ...
        '     source = dsp.ColoredNoise(''OutputDataType'',targetDataType);'                       , ...
        '     inputBuffer = dsp.AsyncBuffer;'                                                      , ...
        '     for ii = 1:10'                                                                       , ...
        '         audioIn = source();'                                                             , ...
        '         write(inputBuffer,audioIn);'                                                     , ...
sprintf('         while inputBuffer.NumUnreadSamples > %d',numel(obj.Window)-obj.OverlapLength)    , ...
sprintf('             x = read(inputBuffer,%d,%d);',numel(obj.Window),obj.OverlapLength)           , ...
sprintf('             featureVector = %s_mex(x);',functionName)                                    , ...
        '             % ... do something with featureVector ...'                                   , ...
        '         end'                                                                             , ...
        '      end'                                                                                , ...
        };
    
end
functionGenerator.SeeAlso = {'audioFeatureExtractor',' dsp.AsyncBuffer',' codegen'};
functionGenerator.RCSRevisionAndDate = false;
functionGenerator.TimeStampInHeader = false;
functionGenerator.EndOfFileMarker = false;

% Add time stamp (with time zone) and coder compatibility keyword
t = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');
functionGenerator.addCode('%%   Generated by audioFeatureExtractor on %s',string(t));
functionGenerator.addCode('%#codegen');
functionGenerator.addCode();
end