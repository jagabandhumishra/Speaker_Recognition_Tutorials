classdef audioFeatureExtractor < handle & ...
        matlab.mixin.CustomDisplay & ...
        matlab.mixin.SetGet & ...
        matlab.mixin.Copyable
%audioFeatureExtractor Streamline audio feature extraction
%
% aFE = audioFeatureExtractor() creates an audioFeatureExtractor object
% with default property values.
%
% aFE = audioFeatureExtractor(Name,Value) specifies nondefault property
% values of the audioFeatureExtractor object.
%
% FEATURES = extract(aFE,audioIn) extracts features from the audio input.
% FEATURES is an L-by-M-by-N array, where
%         L - Number of feature vectors (hops)
%         M - Number of features extracted per analysis window
%         N - Number of channels
%
% audioFeatureExtractor Properties:
%
% General Parameters
% Window                  - Analysis window
% OverlapLength           - Number of samples overlapped between windows
% FFTLength               - FFT length
% SampleRate              - Input sample rate (Hz)
% SpectralDescriptorInput - Input to spectral descriptors
% FeatureVectorLength     - Number of features output from extract (read-only)
%
% Features to Extract (true/false)
% linearSpectrum       - Extract linear spectrum
% melSpectrum          - Extract mel spectrum
% barkSpectrum         - Extract Bark spectrum
% erbSpectrum          - Extract ERB spectrum
% mfcc                 - Extract mfcc
% mfccDelta            - Extract delta mfcc
% mfccDeltaDelta       - Extract delta-delta mfcc
% gtcc                 - Extract gtcc
% gtccDelta            - Extract delta gtcc
% gtccDeltaDelta       - Extract delta-delta gtcc
% spectralCentroid     - Extract spectral centroid
% spectralCrest        - Extract spectral crest
% spectralDecrease     - Extract spectral decrease
% spectralEntropy      - Extract spectral entropy
% spectralFlatness     - Extract spectral flatness
% spectralFlux         - Extract spectral flux
% spectralKurtosis     - Extract spectral kurtosis
% spectralRolloffPoint - Extract spectral rolloff point
% spectralSkewness     - Extract spectral skewness
% spectralSlope        - Extract spectral slope
% spectralSpread       - Extract spectral spread
% pitch                - Extract pitch
% harmonicRatio        - Extract harmonic ratio
% zerocrossrate        - Extract zero-crossing rate
% shortTimeEnergy      - Extract short-time energy
%
% audioFeatureExtractor Methods:
%
% extract                - Extract features
% info                   - Output mapping and individual feature extractor
%                          parameters
% setExtractorParameters - Set nondefault parameter values for individual
%                          feature extractors
% generateMATLABFunction - Generate a MATLAB function from an
%                          audioFeatureExtractor object. The function is
%                          compatible with C/C++ code generation.
%
% Example 1:
%   % Extract MFCC, delta MFCC, delta-delta MFCC, pitch, and spectral
%   % centroid from an audio signal. Use a 30 ms analysis window with 20 ms
%   % overlap. Plot the spectral centroid over time.
%
%   [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
%
%   aFE = audioFeatureExtractor(SampleRate=fs, ...
%             Window=hamming(round(0.03*fs),"periodic"), ...
%             OverlapLength=round(0.02*fs), ...
%             mfcc=true, ...
%             mfccDelta=true, ...
%             mfccDeltaDelta=true, ...
%             pitch=true, ...
%             spectralCentroid=true);
%
%   features = extract(aFE,audioIn);
%
%   idx = info(aFE);
%   t = linspace(0,size(audioIn,1)/fs,size(features,1));
%   plot(t,features(:,idx.spectralCentroid))
%   title('Spectral Centroid')
%   xlabel('Time (s)')
%   ylabel('Frequency (Hz)')
%
% Example 2:
%   % Extract the melSpectrum, barkSpectrum, erbSpectrum, and
%   % linearSpectrum from audio samples included in Audio Toolbox.
%   aFE = audioFeatureExtractor(melSpectrum=true,barkSpectrum=true, ...
%          erbSpectrum=true,linearSpectrum=true,SampleRate=44.1e3);
%
%   folder = fullfile(matlabroot,'toolbox','audio','samples');
%   ads = audioDatastore(folder);
%   keepFile = cellfun(@(x)contains(x,'44p1'),ads.Files);
%   ads = subset(ads,keepFile);
%   adsTall = tall(ads);
%
%   specsTall = cellfun(@(x)extract(aFE,x),adsTall,"UniformOutput",false);
%   specs = gather(specsTall);
%
%   numFiles = numel(specs)
%   [numHopsFile1,numFeaturesFile1,numChannelsFile1] = size(specs{1})
%
% See also AUDIODATASTORE, AUDIODATAAUGMENTER
    
    % Copyright 2019-2021 The MathWorks, Inc.
    
    properties
        %WINDOW Analysis window
        % Specify the analysis window as a numeric vector with more than three
        % elements. The default is hamming(1024,"periodic").
        Window = hamming(1024,"periodic")
        %OVERLAPLENGTH Number of samples overlapped between windows
        % Specify the number of samples overlapped between analysis windows as a
        % positive scalar less than the analysis window length. The default is 512.
        OverlapLength = 512
        %SAMPLERATE Sample rate (Hz)
        % Specify sample rate as a positive scalar in Hz. The default is 44100 Hz.
        SampleRate = 44.1e3;
        %FFTLENGTH FFT length
        % Specify the DFT length as a positive scalar integer. The default of this
        % property is [], which means that the DFT length is equal to the length
        % of the Window property.
        FFTLength = []
        %SPECTRALDESCRIPTORINPUT Input to spectral descriptors
        % Specify the input to the low-level spectral shape
        % descriptors as "linearSpectrum", "melSpectrum", "barkSpectrum", or
        % "erbSpectrum". The default is "linearSpectrum".
        SpectralDescriptorInput (1,:) char {mustBeMember(SpectralDescriptorInput,{'linearSpectrum','melSpectrum','barkSpectrum','erbSpectrum'})} = 'linearSpectrum';
        %LINEARSPECTRUM Extract linear spectrum
        % Extract the linear spectrum, specified as true or false. If
        % linearSpectrum is true, then the object extracts the one-sided linear
        % spectrum and appends it to the features returned. The default is false.
        linearSpectrum = false
        %MELSPECTRUM Extract mel spectrum
        % Extract the mel spectrum, specified as true or false. If melSpectrum is
        % true, then the object extracts the mel spectrum and appends it to the
        % features returned. The default is false.
        %
        % The mel filter bank is designed using designAuditoryFilterBank. You can
        % configure the mel spectrum extraction using setExtractorParameters.
        %
        % See also DESIGNAUDITORYFILTERBANK, MELSPECTROGRAM, SETEXTRACTORPARAMETERS
        melSpectrum = false
        %BARKSPECTRUM Extract Bark spectrum
        % Extract the Bark spectrum, specified as true or false. If barkSpectrum is
        % true, then the object extracts the Bark spectrum and appends it to the
        % features returned. The default is false.
        %
        % The Bark filter bank is designed using designAuditoryFilterBank. You can
        % configure the Bark spectrum extraction using setExtractorParameters.
        %
        % See also DESIGNAUDITORYFILTERBANK, SETEXTRACTORPARAMETERS
        barkSpectrum = false
        %ERBSPECTRUM Extract ERB spectrum
        % Extract the ERB spectrum, specified as true or false. If erbSpectrum is
        % true, then the object extracts the ERB spectrum and appends it to the
        % features returned. The default is false.
        %
        % You can configure the ERB spectrum extraction using setExtractorParameters.
        %
        % See also DESIGNAUDITORYFILTERBANK, SETEXTRACTORPARAMETERS
        erbSpectrum = false
        %MFCC Extract MFCC
        % Extract the mfcc, specified as true or false. If mfcc is true, then the
        % object extracts the mfcc and appends it to the features returned. The
        % default is false.
        %
        % You can configure the MFCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        mfcc = false
        %MFCCDELTA Extract delta mfcc
        % Extract the delta mfcc, specified as true or false. If mfccDelta is true,
        % then the object extracts the delta mfcc and appends it to the features
        % returned. The default is false.
        %
        % You can configure the MFCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        mfccDelta = false
        %MFCCDELTADELTA Extract delta-delta MFCC
        % Extract the delta-delta MFCC, specified as true or false. If
        % mfccDeltaDelta is true, then the object extracts the delta-delta mfcc and
        % appends it to the features returned. The default is false.
        %
        % You can configure the MFCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        mfccDeltaDelta = false
        %GTCC Extract GTCC
        % Extract the GTCC, specified as true or false. If gtcc is true, then the
        % object extracts the GTCC and appends it to the features returned. The
        % default is false.
        %
        % You can configure the GTCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        gtcc = false
        %GTCCDELTA Extract delta GTCC
        % Extract the delta GTCC, specified as true or false. If gtccDelta is true,
        % then the object extracts the delta GTCC and appends it to the features
        % returned. The default is false.
        %
        % You can configure the GTCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        gtccDelta = false
        %GTCCDELTADELTA Extract delta-delta GTCC
        % Extract the delta-delta GTCC, specified as true or false. If
        % gtccDeltaDelta is true, then the object extracts the delta-delta gtcc and
        % appends it to the features returned. The default is false.
        %
        % You can configure the GTCC feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        gtccDeltaDelta = false
        %SPECTRALCENTROID Extract spectral centroid
        % Extract the spectral centroid, specified as true or false. If
        % spectralCentroid is true, then the object extracts the spectral centroid
        % and appends it to the features returned. The default is false.
        spectralCentroid = false
        %SPECTRALCREST Extract spectral crest
        % Extract the spectral crest, specified as true or false. If spectralCrest
        % is true, then the object extracts the spectral crest and appends it to
        % the features returned. The default is false.
        spectralCrest = false
        %SPECTRALDECREASE Extract spectral decrease
        % Extract the spectral decrease, specified as true or false. If
        % spectralDecrease is true, then the object extracts the spectral decrease and
        % appends it to the features returned. The default is false.
        spectralDecrease = false
        %SPECTRALENTROPY Extract spectral entropy
        % Extract the spectral entropy, specified as true or false. If
        % spectralEntropy is true, then the object extracts the spectral entropy
        % and appends it to the features returned. The default is false.
        spectralEntropy = false
        %SPECTRALFLATNESS Extract spectral flatness
        % Extract the spectral flatness, specified as true or false. If
        % spectralFlatness is true, then the object extracts the spectral flatness
        % and appends it to the features returned. The default is false.
        spectralFlatness = false
        %SPECTRALFLUX Extract spectral flux
        % Extract the spectral flux, specified as true or false. If spectralFlux is
        % true, then the object extracts the spectral flux and appends it to the
        % features returned. The default is false.
        %
        % You can configure the spectral flux feature extraction using
        % setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        spectralFlux = false
        %SPECTRALKURTOSIS Extract spectral kurtosis
        % Extract the spectral kurtosis, specified as true or false. If
        % spectralKurtosis is true, then the object extracts the spectral kurtosis
        % and appends it to the features returned. The default is false.
        spectralKurtosis = false
        %SPECTRALROLLOFFPOINT Extract spectral rolloff point
        % Extract the spectral rolloff point, specified as true or false. If
        % spectralRolloffPoint is true, then the object extracts the spectral
        % rolloff point and appends it to the features returned. The default is
        % false.
        %
        % You can configure the spectral rolloff point feature extraction using
        % setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        spectralRolloffPoint = false
        %SPECTRALSKEWNESS Extract spectral skewness
        % Extract the spectral skewness, specified as true or false. If
        % spectralSkewness is true, then the object extracts the spectral skewness
        % and appends it to the features returned. The default is false.
        spectralSkewness = false
        %SPECTRALSLOPE Extract spectral slope
        % Extract the spectral slope, specified as true or false. If spectralSlope
        % is true, then the object extracts the spectral slope and appends it to
        % the features returned. The default is false.
        spectralSlope = false
        %SPECTRALSPREAD Extract spectral spread
        % Extract the spectral spread, specified as true or false. If
        % spectralSpread is true, then the object extracts the spectral spread and
        % appends it to the features returned. The default is false.
        spectralSpread = false
        %PITCH Extract pitch
        % Extract the pitch, specified as true or false. If pitch is true, then the
        % object extracts the pitch and appends it to the features returned. The
        % default is false.
        %
        % You can configure the pitch feature extraction using setExtractorParameters.
        %
        % See also SETEXTRACTORPARAMETERS
        pitch = false
        %HARMONICRATIO Extract harmonic ratio
        % Extract the harmonic ratio, specified as true or false. If harmonicRatio
        % is true, then the object extracts the harmonic ratio and appends it to
        % the features returned. The default is false.
        harmonicRatio = false
        %ZEROCROSSRATE Extract zero-crossing rate
        % Extract the zero-crossing rate, specified as true or false. If
        % zerocrossrate is true, then the object extracts the zero-crossing
        % rate and appends it to the features returned. The default is
        % false.
        %
        % You can configure the zero-crossing rate extraction using
        % setExtractorParameters.
        %
        % See also setExtractorParameters
        zerocrossrate = false
        %SHORTTIMEENERGY Extract short-time energy
        % Extract the short-time energy, specified as true or false. If
        % shortTimeEnergy is true, then the object extracts the energy per
        % window and appends it to the features returned. The default is
        % false.
        shortTimeEnergy = false
    end
    properties(Access = private)
        % Properties to hold user-specified feature extractor parameters
        % (specified using setExtractorParameters).
        pmfccUserSpecifiedParams                 = {}
        pgtccUserSpecifiedParams                 = {}
        pspectralFluxUserSpecifiedParams         = {}
        pspectralRolloffPointUserSpecifiedParams = {}
        plinearSpectrumUserSpecifiedParams       = {}
        pmelSpectrumUserSpecifiedParams          = {}
        pbarkSpectrumUserSpecifiedParams         = {}
        perbSpectrumUserSpecifiedParams          = {}
        ppitchUserSpecifiedParams                = {}
        pzerocrossrateUserSpecifiedParams        = {}

        plinearSpectrumType = "power";
        pmelSpectrumType    = "power";
        pbarkSpectrumType   = "power";
        perbSpectrumType    = "power";
        
        plinearSpectrumWinNorm = true;
        pmelSpectrumWinNorm    = true;
        pbarkSpectrumWinNorm   = true;
        perbSpectrumWinNorm    = true;
    end
    properties(Transient,Access = private)
        % Flag for setup
        pIsInitialized = false
        pPrototype = zeros(0,'double');
        
        % Convenience properties to navigate the feature extraction pipeline
        % These properties are all reevaluated when setup is called. Setup is
        % called when the object is created, loaded, or if the input data type has
        % changed.
        pPipelineParameters
        pExtractorParameters
        pOutputIndex
        pOutputIndexReduced
        pUseSpectrum
        pUseHalfSpectrum
        pUsePowerSpectrum
        pUseMagnitudeSpectrum
        pExtractSpectralDescriptor
        pCalculateGTCC
        pCalculateMFCC
        pCalculateBarkSpectrum
        pCalculateMelSpectrum
        pCalculateLinearSpectrum
        pCalculateERBSpectrum
        pUseWindowNormalizationMagnitude
        pUseWindowNormalizationPower
    end
    properties (Transient,Dependent,GetAccess = public,SetAccess = private)
        %FEATUREVECTORLENGTH Number of features output from extract (read-only)
        % The total number of features output from extract for the current
        % object configuration. FeatureVectorLength is equal to the second
        % dimension of the output from extract.
        FeatureVectorLength
    end
    properties (Transient,Dependent,Access = private)
        pFeaturesToExtract
        pFFTLength
    end
    % Static Methods - Load Object
    methods(Static)
        function obj = loadobj(s)
            if isstruct(s)
                % remove dependent properties from list
                if isfield(s,'pFeaturesToExtract')
                    s = rmfield(s,'pFeaturesToExtract');
                end
                if isfield(s,'pFFTLength')
                    s = rmfield(s,'pFFTLength');
                end

                % initalize object
                obj = audioFeatureExtractor;

                % remove fields not in current object (for forwards
                % compatibility)
                fn = fieldnames(s);
                for ii = 1:numel(fn)
                    if ~isprop(obj,fn{ii})
                        s = rmfield(s,fn{ii});
                    end
                end

                % set properties
                set(obj,s);
            else
                obj = s;
            end
            obj.pIsInitialized = false;
        end
    end
    
    % Public Methods
    methods
        generateMATLABFunction(obj,~,~,~)
    end
    methods
        function obj = audioFeatureExtractor(varargin)
            if nargin > 0
                defaultWindow = obj.Window;
                defaultOverlapLength = obj.OverlapLength;
                defaultSampleRate = obj.SampleRate;
                defaultFFTLength = obj.FFTLength;
                defaultSpectralDescriptorInput = obj.SpectralDescriptorInput;

                inpP = inputParser();
                addParameter(inpP, 'Window', defaultWindow);
                addParameter(inpP, 'OverlapLength', defaultOverlapLength);
                addParameter(inpP, 'SampleRate', defaultSampleRate);
                addParameter(inpP, 'FFTLength', defaultFFTLength);
                addParameter(inpP, 'SpectralDescriptorInput', defaultSpectralDescriptorInput);

                featuresToExtract = obj.pFeaturesToExtract;
                featureNames = fieldnames(featuresToExtract);
                for idx = 1:numel(featureNames)
                    addParameter(inpP, featureNames{idx}, false);
                end
                parse(inpP, varargin{:});
                parsedStruct = inpP.Results;
                set(obj,parsedStruct)
            end
        end
        function setExtractorParameters(obj,propname,varargin)
            %setExtractorParameters Set nondefault parameter values of feature extractor
            % setExtractorParameters(aFE,FEATURENAME,PARAMS) specifies parameters used to
            % extract FEATURENAME. Specify the feature name as a string or character
            % vector. Specify PARAMS as comma-separated name-value pairs or as a
            % struct.
            %
            % setExtractorParameters(aFE,FEATURENAME) returns the parameters used to
            % extract FEATURENAME to default values.
            %
            % Example 1:
            %   % Extract pitch using the LHS method.
            %   [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
            %
            %   aFE = audioFeatureExtractor(SampleRate=fs,pitch=true);
            %   setExtractorParameters(aFE,"pitch",Method="LHS");
            %
            %   features = extract(aFE,audioIn);
            %
            % Example 2:
            %   % Extract the spectral rolloff point with a threshold of 0.8 and a
            %   % 20-band mel spectrum.
            %   [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
            %
            %   aFE = audioFeatureExtractor(SampleRate=fs,melSpectrum=true, ...
            %                               spectralRolloffPoint=true);
            %   [~,params] = info(aFE);
            %   params.melSpectrum.NumBands = 20;
            %   params.spectralRolloffPoint.Threshold = 0.8;
            %   setExtractorParameters(aFE,"melSpectrum",params.melSpectrum);
            %   setExtractorParameters(aFE,"spectralRolloffPoint", ...
            %       params.spectralRolloffPoint);
            %
            %   features = extract(aFE,audioIn);
            %
            % See also INFO
            propname = validatestring(propname,["mfcc","gtcc", ...
                "spectralFlux","spectralRolloffPoint", ...
                "melSpectrum","barkSpectrum","linearSpectrum","erbSpectrum", ...
                "pitch", "harmonicRatio","zerocrossrate"],"setExtractorParameters");
            
            winLength = numel(obj.Window);
            x = ones(winLength,1);
            fs = obj.SampleRate;
            
            % If input was a struct, unpack it to a name-value cell
            numin = numel(varargin);
            if numin==1 && isscalar([varargin{:}]) && isstruct(varargin{:})
                varargin = namedargs2cell(varargin{:});
                numin = numel(varargin);
            end
            if numin~=0
                [varargin{:}] = gather(varargin{:});
            end
            userSpecParams = struct(varargin{:});
            if numin~=0
                previousUserSpecParams = obj.getFeatureExtractUserSpecifiedParams(propname);
                userSpecParams = obj.mergeStructs(previousUserSpecParams,userSpecParams);
            end
            
            if isempty(fieldnames(userSpecParams))
                % If the user-specified params is empty, set the current config
                % back to default
                currentConfig = getFeatureExtractorDefaultParams(obj,propname);
            else
                % Get the previously set configuration of the feature extractor.
                currentConfig = getFeatureExtractorParams(obj,propname);
            end

            % Add the obsolete Normalization parameter to the current
            % config
            if any(strcmp(propname,{'melSpectrum','barkSpectrum','erbSpectrum'}))
                currentConfig.Normalization = currentConfig.FilterBankNormalization;
            end

            % Parse all parameters
            if nargin > 0
                inpP = inputParser();
                paramNames = fieldnames(currentConfig);
                for ii = 1:numel(paramNames)
                    addParameter(inpP,paramNames{ii},currentConfig.(paramNames{ii}))
                end
                parse(inpP, varargin{:});
                params = inpP.Results;

                % Convert the case-insensitive and partial matching
                % parameters to the full and case-sensitive parameters.
                validfn = fieldnames(params);
                fn = fieldnames(userSpecParams);
                for ii = 1:numel(fn)
                    oldField = fn{ii};
                    newField = validatestring(oldField,validfn);
                    if ~isequal(oldField,newField)
                        userSpecParams.(newField) = userSpecParams.(oldField);
                        userSpecParams = rmfield(userSpecParams,oldField);
                    end
                end
            end

            % Verify and remove the obsolete parameters
            [params,userSpecParams] = parseObsoleteParams(obj,propname,params,userSpecParams);

            switch propname
                case "mfcc"
                    validateattributes(params.NumCoeffs,{'numeric'}, ...
                        {'nonempty','integer','scalar','real','>',1}, ...
                        'mfcc','NumCoeffs');
                    params.Rectification = validatestring(params.Rectification,["cubic-root","log"],'mfcc','Rectification');
                    if isfield(userSpecParams,'Rectification')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.Rectification = params.Rectification;
                    end
                    validateattributes(params.DeltaWindowLength,{'numeric'},...
                        {'nonempty','scalar','integer','positive','odd','>',1}, ...
                        'mfcc','DeltaWindowLength');

                    obj.pmfccUserSpecifiedParams = userSpecParams;

                case "gtcc"
                    validateattributes(params.NumCoeffs,{'numeric'}, ...
                        {'nonempty','integer','scalar','real','>',1}, ...
                        'gtcc','NumCoeffs');
                    params.Rectification = validatestring(params.Rectification,["cubic-root","log"],'gtcc','Rectification');
                    if isfield(userSpecParams,'Rectification')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.Rectification = params.Rectification;
                    end
                    validateattributes(params.DeltaWindowLength,{'numeric'},...
                        {'nonempty','scalar','integer','positive','odd','>',1}, ...
                        'gtcc','DeltaWindowLength');
                    
                    obj.pgtccUserSpecifiedParams = userSpecParams;

                case "spectralFlux"
                    validateattributes(params.NormType,{'numeric'}, ...
                        {'nonempty','scalar','integer','>',0,'<',3})
                    
                    obj.pspectralFluxUserSpecifiedParams = userSpecParams;
                    
                case "spectralRolloffPoint"
                    validateattributes(params.Threshold,{'single','double'}, ...
                        {'nonempty','scalar','>',0,'<',1,'real'}, ...
                        mfilename,'Threshold');
                    
                    obj.pspectralRolloffPointUserSpecifiedParams = userSpecParams;
                    
                case "linearSpectrum"
                    params.SpectrumType = validatestring(params.SpectrumType,["power","magnitude"],"linearSpectrum","SpectrumType");
                    if isfield(userSpecParams,'SpectrumType')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.SpectrumType = params.SpectrumType;
                    end

                    validateattributes(params.WindowNormalization,{'numeric','logical'}, ...
                        {'nonempty','scalar','real'}, ...
                        'setExtractorParameters','WindowNormalization');
                    
                    obj.plinearSpectrumType = params.SpectrumType;
                    obj.plinearSpectrumWinNorm = params.WindowNormalization;
                    
                    validateattributes(params.FrequencyRange,{'single','double'}, ...
                        {"nonempty","increasing","nonnegative","row","ncols",2,"real"}, ...
                        "linearSpectrum","FrequencyRange")
                    validateattributes(params.FrequencyRange(2),{'single','double'}, ...
                        {'<=',obj.SampleRate/2}, ...
                        'linearSpectrum','FrequencyRange')
                    
                    obj.plinearSpectrumUserSpecifiedParams = userSpecParams;
                    
                case "melSpectrum"
                    params.SpectrumType = validatestring(params.SpectrumType,["power","magnitude"],"melSpectrum","SpectrumType");
                    if isfield(userSpecParams,'SpectrumType')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.SpectrumType = params.SpectrumType;
                    end
                    validateattributes(params.WindowNormalization,{'numeric','logical'}, ...
                        {'nonempty','scalar','real'}, ...
                        'setExtractorParameters','WindowNormalization');
                    
                    obj.pmelSpectrumType = params.SpectrumType;
                    obj.pmelSpectrumWinNorm = params.WindowNormalization;
                    filtBankNorm = validatestring(params.FilterBankNormalization,["area","bandwidth","none"],"melSpectrum","FilterBankNormalization");
                    params.FilterBankDesignDomain = validatestring(params.FilterBankDesignDomain,["linear","warped"],"melSpectrum","FilterBankDesignDomain");
                    params = rmfield(params,["SpectrumType","WindowNormalization","FilterBankNormalization"]);
                    
                    nv = namedargs2cell(params);
                    designAuditoryFilterBank(fs,"FFTLength",obj.pFFTLength, ...
                        "FrequencyScale","mel",'Normalization',filtBankNorm, ...
                        nv{:});
                    
                    if isfield(userSpecParams,'FilterBankDesignDomain')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.FilterBankDesignDomain = params.FilterBankDesignDomain;
                    end
                    if isfield(userSpecParams,'FilterBankNormalization')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.FilterBankNormalization = filtBankNorm;
                    end
                    obj.pmelSpectrumUserSpecifiedParams = userSpecParams;
                    
                case "barkSpectrum"
                    params.SpectrumType = validatestring(params.SpectrumType,["power","magnitude"],"barkSpectrum","SpectrumType");
                    if isfield(userSpecParams,'SpectrumType')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.SpectrumType = params.SpectrumType;
                    end
                    validateattributes(params.WindowNormalization,{'numeric','logical'}, ...
                        {'nonempty','scalar','real'}, ...
                        'setExtractorParameters','WindowNormalization');
                    
                    obj.pbarkSpectrumType = params.SpectrumType;
                    obj.pbarkSpectrumWinNorm = params.WindowNormalization;
                    filtBankNorm = validatestring(params.FilterBankNormalization,["area","bandwidth","none"],"barkSpectrum","FilterBankNormalization");
                    params.FilterBankDesignDomain = validatestring(params.FilterBankDesignDomain,["linear","warped"],"barkSpectrum","FilterBankDesignDomain");
                    params = rmfield(params,["SpectrumType","WindowNormalization","FilterBankNormalization"]);

                    nv = namedargs2cell(params);
                    designAuditoryFilterBank(fs,"FFTLength",obj.pFFTLength, ...
                        "FrequencyScale","bark",'Normalization',filtBankNorm, ...
                        nv{:});

                    if isfield(userSpecParams,'FilterBankDesignDomain')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.FilterBankDesignDomain = params.FilterBankDesignDomain;
                    end
                    if isfield(userSpecParams,'FilterBankNormalization')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.FilterBankNormalization = filtBankNorm;
                    end
                    obj.pbarkSpectrumUserSpecifiedParams = userSpecParams;
                    
                case "erbSpectrum"
                    params.SpectrumType = validatestring(params.SpectrumType,["power","magnitude"],"erbSpectrum","SpectrumType");
                    if isfield(userSpecParams,'SpectrumType')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.SpectrumType = params.SpectrumType;
                    end
                    validateattributes(params.WindowNormalization,{'numeric','logical'}, ...
                        {'nonempty','scalar','real'}, ...
                        'setExtractorParameters','WindowNormalization');
                    
                    obj.perbSpectrumType = params.SpectrumType;
                    obj.perbSpectrumWinNorm = params.WindowNormalization;
                    filtBankNorm = validatestring(params.FilterBankNormalization,["area","bandwidth","none"],"erbSpectrum","FilterBankNormalization");
                    params = rmfield(params,["SpectrumType","WindowNormalization","FilterBankNormalization"]);
                    
                    nv = namedargs2cell(params);
                    designAuditoryFilterBank(fs,"FFTLength",obj.pFFTLength, ...
                        "FrequencyScale","erb",'Normalization',filtBankNorm, ...
                        nv{:});

                    if isfield(userSpecParams,'FilterBankNormalization')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.FilterBankNormalization = filtBankNorm;
                    end
                    obj.perbSpectrumUserSpecifiedParams = userSpecParams;
                    
                case "pitch"
                    params.Method = validatestring(params.Method,["PEF","NCF","CEP","LHS","SRH"],"pitch","Method");
                    if isfield(userSpecParams,'Method')
                        % Convert partially matched/case insensitive
                        % parameter.
                        userSpecParams.Method = params.Method;
                    end
                    nv = namedargs2cell(params);
                    [~] = pitch(x,fs,"WindowLength",numel(obj.Window), ...
                        "OverlapLength",obj.OverlapLength, ...
                        nv{:});
                    
                    obj.ppitchUserSpecifiedParams = userSpecParams;

                case "zerocrossrate"
                    nv = namedargs2cell(params);
                    suppressplot = zerocrossrate(x,"WindowLength",numel(obj.Window), ...
                        "OverlapLength",obj.OverlapLength, ...
                        nv{:}); %#ok<NASGU> 

                    obj.pzerocrossrateUserSpecifiedParams = userSpecParams;
            end
            obj.pIsInitialized = false;
        end
        function featureVector = extract(obj,x)
            %EXTRACT Extract features
            % features = extract(aFE,x) returns an array or cell array containing
            % features of the audio input, x. The output depends on properties of aFE.
            %
            % Example:
            %   % Extract the spectral centroid, spectral kurtosis, and pitch of an
            %   % audio signal. Plot their normalized values over time.
            %   [audioIn,fs] = audioread("Counting-16-44p1-mono-15secs.wav");
            %
            %    aFE = audioFeatureExtractor(SampleRate=fs, ...
            %                                spectralCentroid=true, ...
            %                                spectralKurtosis=true, ...
            %                                pitch=true);
            %
            %    features = extract(aFE,audioIn);
            %    features = (features - mean(features,1))./std(features,[],1);
            %    idx = info(aFE);
            %
            %    subplot(2,1,1)
            %    t = linspace(0,15,size(audioIn,1));
            %    plot(t,audioIn)
            %
            %    subplot(2,1,2)
            %    t = linspace(0,15,size(features,1));
            %    plot(t,features(:,idx.spectralCentroid), ...
            %         t,features(:,idx.spectralKurtosis), ...
            %         t,features(:,idx.pitch));
            %    legend("Spectral Centroid","Spectral Kurtosis","Pitch");
            %
            
            %#ok<*CPROPLC>
            
            validateattributes(x,{'single','double'},{'2d','real'},'extract','audioIn')
            
            DT = underlyingType(x);
            OT = class(x);
            pDT = underlyingType(obj.pPrototype);
            pOT = class(obj.pPrototype);
            if ~obj.pIsInitialized || ~strcmp(DT,pDT) || ~strcmp(OT,pOT)
                prototype = zeros(0,'like',x);
                setup(obj,prototype)
            end
            
            featureVector = obj.pPrototype;
            
            [numSamp,numChan] = size(x);
            if numSamp >= obj.pPipelineParameters.WindowLength
                linearSpectrumF       = featureVector;
                melSpectrumF          = featureVector;
                barkSpectrumF         = featureVector;
                erbSpectrumF          = featureVector;
                mfccF                 = featureVector;
                mfccDeltaF            = featureVector;
                mfccDeltaDeltaF       = featureVector;
                gtccF                 = featureVector;
                gtccDeltaF            = featureVector;
                gtccDeltaDeltaF       = featureVector;
                spectralCentroidF     = featureVector;
                spectralCrestF        = featureVector;
                spectralDecreaseF     = featureVector;
                spectralEntropyF      = featureVector;
                spectralFlatnessF     = featureVector;
                spectralFluxF         = featureVector;
                spectralKurtosisF     = featureVector;
                spectralRolloffPointF = featureVector;
                spectralSkewnessF     = featureVector;
                spectralSlopeF        = featureVector;
                spectralSpreadF       = featureVector;
                pitchF                = featureVector;
                harmonicRatioF        = featureVector;
                zerocrossrateF        = featureVector;
                shortTimeEnergyF      = featureVector;

                xb = audio.internal.buffer(x,obj.pPipelineParameters.WindowLength,obj.pPipelineParameters.HopLength);
                xbw = xb.*obj.pPipelineParameters.Window;
                numHops = size(xbw,2)/numChan;
                
                if obj.pUseSpectrum
                    Z = fft(xbw,obj.pPipelineParameters.FFTLength,1);
                    if isreal(Z)
                        Z = complex(Z);
                    end
                    if isa(x,'gpuArray')
                        Z = head(Z,obj.pPipelineParameters.OneSidedSpectrumBins(end));
                    else
                        Z = Z(obj.pPipelineParameters.OneSidedSpectrumBinsLogical,:,:);
                    end
                    if obj.pUsePowerSpectrum
                        Zpower = real(Z.*conj(Z));
                        if obj.pUseMagnitudeSpectrum
                            Zmagnitude = sqrt(Zpower);
                        end
                    elseif obj.pUseMagnitudeSpectrum
                        Zmagnitude = abs(Z);
                    end
                end
                
                if obj.pCalculateLinearSpectrum
                    if obj.pPipelineParameters.linearSpectrum.IsPower
                        if obj.pPipelineParameters.linearSpectrum.FullSpectrum
                            linSpec = Zpower*obj.pPipelineParameters.linearSpectrum.NormalizationFactor;
                        else
                            linSpec = Zpower(obj.pPipelineParameters.linearSpectrum.FrequencyBinsLogical,:)*obj.pPipelineParameters.linearSpectrum.NormalizationFactor;
                        end
                    else
                        if obj.pPipelineParameters.linearSpectrum.FullSpectrum
                            linSpec = Zmagnitude*obj.pPipelineParameters.linearSpectrum.NormalizationFactor;
                        else
                            linSpec = Zmagnitude(obj.pPipelineParameters.linearSpectrum.FrequencyBinsLogical,:)*obj.pPipelineParameters.linearSpectrum.NormalizationFactor;
                        end
                    end
                    if obj.pPipelineParameters.linearSpectrum.AdjustBins
                        linSpec = linSpec.*obj.pPipelineParameters.linearSpectrum.AdjustBinsVector;
                    end
                    linSpec = reshape(linSpec,[],numHops,numChan);
                    if obj.linearSpectrum
                        linearSpectrumF = permute(linSpec,[2,1,3]);
                    end
                end
                
                if obj.pCalculateMelSpectrum
                    numBands = size(obj.pPipelineParameters.melSpectrum.FilterBank,1);
                    if obj.pPipelineParameters.melSpectrum.IsPower
                        melSpec = reshape(obj.pPipelineParameters.melSpectrum.FilterBank*Zpower,numBands,numHops,numChan);
                    else
                        melSpec = reshape(obj.pPipelineParameters.melSpectrum.FilterBank*Zmagnitude,numBands,numHops,numChan);
                    end
                    if obj.melSpectrum
                        melSpectrumF = permute(melSpec,[2,1,3]);
                    end
                end
                
                if obj.pCalculateBarkSpectrum
                    numBands = size(obj.pPipelineParameters.barkSpectrum.FilterBank,1);
                    if obj.pPipelineParameters.barkSpectrum.IsPower
                        barkSpec = reshape(obj.pPipelineParameters.barkSpectrum.FilterBank*Zpower,numBands,numHops,numChan);
                    else
                        barkSpec = reshape(obj.pPipelineParameters.barkSpectrum.FilterBank*Zmagnitude,numBands,numHops,numChan);
                    end
                    if obj.barkSpectrum
                        barkSpectrumF = permute(barkSpec,[2,1,3]);
                    end
                end
                
                if obj.pCalculateERBSpectrum
                    numBands = size(obj.pPipelineParameters.erbSpectrum.FilterBank,1);
                    if obj.pPipelineParameters.erbSpectrum.IsPower
                        erbSpec = reshape(obj.pPipelineParameters.erbSpectrum.FilterBank*Zpower,numBands,numHops,numChan);
                    else
                        erbSpec = reshape(obj.pPipelineParameters.erbSpectrum.FilterBank*Zmagnitude,numBands,numHops,numChan);
                    end
                    if obj.erbSpectrum
                        erbSpectrumF = permute(erbSpec,[2,1,3]);
                    end
                end
                
                if obj.pCalculateMFCC
                    S = melSpec;
                    if strcmp(obj.pPipelineParameters.mfcc.Parameters.Rectification,'log')
                        amin = realmin(DT);
                        if isa(x,'gpuArray')
                            S = arrayfun(@(S,amin)log10(max(S,amin)),S,amin);
                        else
                            S(S==0) = amin;
                            S = log10(S);
                        end
                    elseif  strcmp(obj.pPipelineParameters.mfcc.Parameters.Rectification,'cubic-root')
                        S = S.^(1/3);
                    end
                    [L,M,N] = size(S);
                    S = reshape(S,L,M*N);

                    mfccT = permute(reshape(obj.pPipelineParameters.mfcc.DCTMatrix*S,obj.pPipelineParameters.mfcc.Parameters.NumCoeffs,M,N),[2 1 3]);
                    if obj.mfcc
                        mfccF = mfccT;
                    end
                    if obj.mfccDelta || obj.mfccDeltaDelta
                        mfccDeltaT = filter(obj.pPipelineParameters.mfcc.deltaFilterCoeffs,1,mfccT,[],1);
                        if obj.mfccDelta
                            mfccDeltaF = mfccDeltaT;
                        end
                        if obj.mfccDeltaDelta
                            mfccDeltaDeltaF = filter(obj.pPipelineParameters.mfcc.deltaFilterCoeffs,1,mfccDeltaT,[],1);
                        end
                    end
                end
                
                
                if obj.pCalculateGTCC
                    S = erbSpec;
                    if strcmp(obj.pPipelineParameters.gtcc.Parameters.Rectification,'log')
                        amin = realmin(DT);
                        if isa(x,'gpuArray')
                            S = arrayfun(@(S,amin)log10(max(S,amin)),S,amin);
                        else
                            S(S==0) = amin;
                            S = log10(S);
                        end
                    elseif  strcmp(obj.pPipelineParameters.gtcc.Parameters.Rectification,'cubic-root')
                        S = S.^(1/3);
                    end
                    [L,M,N] = size(S);

                    gtccT = permute(reshape(obj.pPipelineParameters.gtcc.DCTMatrix*reshape(S,L,M*N),obj.pPipelineParameters.gtcc.Parameters.NumCoeffs,M,N),[2 1 3]);
                    if obj.gtcc
                        gtccF = gtccT;
                    end
                    if obj.gtccDelta || obj.gtccDeltaDelta
                        gtccDeltaT = filter(obj.pPipelineParameters.gtcc.deltaFilterCoeffs,1,gtccT,[],1);
                        if obj.gtccDelta
                            gtccDeltaF = gtccDeltaT;
                        end
                        if obj.gtccDeltaDelta
                            gtccDeltaDeltaF = filter(obj.pPipelineParameters.gtcc.deltaFilterCoeffs,1,gtccDeltaT,[],1);
                        end
                    end
                end
                
                if obj.pExtractSpectralDescriptor
                    switch obj.SpectralDescriptorInput
                        case 'melSpectrum'
                            S = melSpec;
                        case 'barkSpectrum'
                            S = barkSpec;
                        case 'erbSpectrum'
                            S = erbSpec;
                        case 'linearSpectrum'
                            S = linSpec;
                    end
                    sizeX1 = size(S,1);
                    f = obj.pPipelineParameters.SpectralDescriptorInput.FrequencyVector(:);
                    if obj.spectralEntropy || obj.spectralCrest || obj.spectralFlatness || ...
                            obj.spectralSlope || obj.spectralCentroid || obj.spectralSpread || obj.spectralSkewness || obj.spectralKurtosis
                        sumX1 = sum(S,1);
                    end
                    if obj.spectralCrest || obj.spectralFlatness || obj.spectralSlope
                        arithmeticMean = sumX1 ./ sizeX1;
                    end
                    if obj.spectralCentroid || obj.spectralSpread || obj.spectralSkewness || obj.spectralKurtosis
                        centroid = sum(S.*f,1) ./ sumX1;
                    end
                    if obj.spectralSpread || obj.spectralSkewness || obj.spectralKurtosis
                        higherMomentTemp = f - centroid;
                        spread = sqrt(sum((higherMomentTemp.^2).*S)./sumX1);
                    end
                    if obj.spectralSkewness || obj.spectralKurtosis
                        higherMomentDenom = (spread.^3).*sumX1;
                        higherMomementNum = (higherMomentTemp.^3).*S;
                    end
                    
                    if obj.spectralCentroid
                        spectralCentroidF = reshape(centroid,[],1,numChan);
                    end
                    if obj.spectralCrest
                        peak = max(S,[],1);
                        crest = peak./arithmeticMean;
                        spectralCrestF = reshape(crest,[],1,numChan);
                    end
                    if obj.spectralDecrease
                        if isa(x,'gpuArray')
                            a = tail(S,sizeX1-1);
                            decrease = sum((a - head(S,1))./(1:sizeX1-1)',1)./sum(a,1);
                        else
                            a = S(2:end,:);
                            decrease = sum((a - S(1,:))./(1:sizeX1-1)',1)./sum(a,1);
                        end
                        spectralDecreaseF = reshape(decrease,[],1,numChan);
                    end
                    if obj.spectralEntropy
                        Xaa = S./repmat(sumX1,sizeX1,1);
                        entropy = -sum(Xaa.*log2(Xaa),1,'omitnan')./log2(sizeX1);
                        spectralEntropyF = reshape(entropy,[],1,numChan);
                    end
                    if obj.spectralFlatness
                        geometricMean = exp(sum(log(S + eps(DT)),1)/sizeX1);
                        flatness = geometricMean./arithmeticMean;
                        spectralFlatnessF = reshape(flatness,[],1,numChan);
                    end
                    if obj.spectralFlux
                        initialCondition = S(:,1,:);
                        fx = vecnorm(diff([initialCondition S],1,2),obj.pPipelineParameters.spectralFlux.Parameters.NormType,1);
                        spectralFluxF = reshape(fx,[],1,numChan);
                    end
                    if obj.spectralKurtosis
                        kurtosisA = sum(higherMomementNum.*higherMomentTemp) ./ (higherMomentDenom.*spread);
                        spectralKurtosisF = reshape(kurtosisA,[],1,numChan);
                    end
                    if obj.spectralRolloffPoint
                        rolloffPoint = spectralRolloffPoint(S,f,'Threshold',obj.pPipelineParameters.spectralRolloffPoint.Parameters.Threshold);
                        spectralRolloffPointF = reshape(rolloffPoint,[],1,numChan);
                    end
                    if obj.spectralSkewness
                        skewness = sum(higherMomementNum) ./ higherMomentDenom;
                        spectralSkewnessF = reshape(skewness,[],1,numChan);
                    end
                    if obj.spectralSlope
                        f_minus_mu_f = f - sum(f,1)./sizeX1;
                        X_minus_mu_X = S - arithmeticMean;
                        slope = sum(X_minus_mu_X.*f_minus_mu_f,1) ./ sum(f_minus_mu_f.^2);
                        spectralSlopeF = reshape(slope,[],1,numChan);
                    end
                    if obj.spectralSpread
                        spectralSpreadF = reshape(spread,[],1,numChan);
                    end
                end
                
                if obj.pitch
                    pitchF = pitch(x,obj.pPipelineParameters.SampleRate,'WindowLength',obj.pPipelineParameters.WindowLength,'OverlapLength',obj.pPipelineParameters.OverlapLength, ...
                        'MedianFilterLength',obj.pPipelineParameters.pitch.Parameters.MedianFilterLength,'Method',obj.pPipelineParameters.pitch.Parameters.Method, ...
                        'Range',obj.pPipelineParameters.pitch.Parameters.Range);
                    pitchF = reshape(pitchF,[],1,numChan);
                end
                
                if obj.harmonicRatio
                    harmonicRatioF = harmonicRatio(xbw,obj.pPipelineParameters.SampleRate,'Window',ones(obj.pPipelineParameters.WindowLength,1),'OverlapLength',0);
                    harmonicRatioF = reshape(harmonicRatioF,[],1,numChan);
                end
                
                if obj.zerocrossrate
                    xb3 = reshape(xb,obj.pPipelineParameters.WindowLength,[],numChan);
                    state = reshape([zeros(1,1,numChan,'like',x),xb3(obj.pPipelineParameters.HopLength,1:end-1,:)],1,[]);
                    zerocrossrateF = zerocrossrate(xb, ...
                        InitialState   = state, ...
                        ZeroPositive   = obj.pPipelineParameters.zerocrossrate.Parameters.ZeroPositive, ...
                        Level          = obj.pPipelineParameters.zerocrossrate.Parameters.Level, ...
                        Method         = obj.pPipelineParameters.zerocrossrate.Parameters.Method, ...
                        Threshold      = obj.pPipelineParameters.zerocrossrate.Parameters.Threshold, ...
                        TransitionEdge = obj.pPipelineParameters.zerocrossrate.Parameters.TransitionEdge);
                    zerocrossrateF = reshape(zerocrossrateF,[],1,numChan);
                end

                if obj.shortTimeEnergy
                    if obj.pUseSpectrum
                        evenLengthFFT = ~rem(obj.pFFTLength,2);
                        if evenLengthFFT
                            shortTimeEnergyF = (2*sum(Zpower,1) - Zpower(1,:) - Zpower(end,:))./obj.pFFTLength;
                        else
                            shortTimeEnergyF = (2*sum(Zpower,1) - Zpower(1,:))./obj.pFFTLength;
                        end
                    else
                        shortTimeEnergyF = sum(xbw.^2,1);
                    end
                    shortTimeEnergyF = reshape(shortTimeEnergyF,[],1,numChan);
                end

                featureVector = horzcat(linearSpectrumF, melSpectrumF, barkSpectrumF, erbSpectrumF, mfccF, mfccDeltaF, ...
                    mfccDeltaDeltaF, gtccF, gtccDeltaF, gtccDeltaDeltaF, spectralCentroidF, spectralCrestF, ...
                    spectralDecreaseF, spectralEntropyF, spectralFlatnessF, spectralFluxF, spectralKurtosisF, spectralRolloffPointF, ...
                    spectralSkewnessF, spectralSlopeF, spectralSpreadF, pitchF, harmonicRatioF, zerocrossrateF, shortTimeEnergyF);
            end
            
        end
        function [idx,params] = info(obj,varargin)
            %INFO Output mapping and individual feature extractor parameters
            % IDX = info(aFE) returns a struct with field names corresponding to
            % enabled feature extractors. The field values correspond to the column
            % indices that the extracted features occupy in the output from extract.
            %
            % IDX = info(aFE,"all") returns a struct with field names corresponding to
            % all available feature extractors. If the feature extractor is disabled,
            % the field value is empty.
            %
            % [IDX,PARAMS] = info(...) returns a second struct, PARAMS. The field
            % names of PARAMS correspond to feature extractors with settable
            % parameters. You can set parameters using setExtractorParameters.
            %
            % Example 1:
            %   % Extract the mel spectrum, mel spectral centroid, and mel spectral
            %   % kurtosis from a random audio signal. Use info to determine which
            %   % columns of the output correspond to which feature, and then plot the
            %   % features separately.
            %
            %   aFE = audioFeatureExtractor(melSpectrum=true, ...
            %             SpectralDescriptorInput='melSpectrum', ...
            %             spectralCentroid=true, ...
            %             spectralKurtosis=true);
            %
            %   features = extract(aFE,2*rand(44.1e3,1)-1);
            %   idx = info(aFE);
            %
            %   figure
            %   surf(features(:,idx.melSpectrum),EdgeColor='none');
            %   title('Mel Spectrum')
            %   figure
            %   plot(features(:,idx.spectralCentroid))
            %   title('Mel Spectral Centroid')
            %   figure
            %   plot(features(:,idx.spectralKurtosis))
            %   title('Mel Spectral Kurtosis')
            %
            %
            % Example 2:
            %   % Extract a 20-band magnitude ERB spectrum from a random signal.
            %
            %   aFE = audioFeatureExtractor(erbSpectrum=true);
            %
            %   [~,params] = info(aFE);
            %   erbParams = params.erbSpectrum;
            %   erbParams.SpectrumType = 'magnitude';
            %   erbParams.NumBands = 20;
            %   setExtractorParameters(aFE,'erbSpectrum',erbParams);
            %
            %   features = extract(aFE,2*rand(1e5,1)-1);
            %
            % See also SETEXTRACTORPARAMETERS
            
            displayAll = false;
            if nargin>1
                validatestring(varargin{:},"all","info","flag",2);
                displayAll = true;
            end
            
            if displayAll
                idxStruct = getOutputIndex(obj);
            else
                [~,idxStruct] = getOutputIndex(obj);
            end
            
            if nargout == 0
                % no output variable, display the output mappings
                featureNames = fieldnames(idxStruct);
                if numel(featureNames) > 0
                    fprintf('   %s\n\n',getString(message('audio:audioFeatureExtractor:OutputMapping')));
                    rightJustified = cellstr(strjust(char(featureNames),'right'));
                    for kdx = 1:numel(featureNames)
                        val = idxStruct.(featureNames{kdx});
                        if isempty(val)
                            fprintf('      %s: <missing>\n',rightJustified{kdx});
                        elseif isscalar(val)
                            fprintf('      %s: %d\n',rightJustified{kdx},val);
                        else
                            fprintf('      %s: %d:%d\n',rightJustified{kdx},val(1),val(end));
                        end
                    end
                end
            elseif nargout == 1
                idx = idxStruct;
            elseif nargout == 2
                idx = idxStruct;
                params = struct( ...
                    "linearSpectrum",       getFeatureExtractorParams(obj,"linearSpectrum"), ...
                    "melSpectrum",          getFeatureExtractorParams(obj,"melSpectrum"), ...
                    "barkSpectrum",         getFeatureExtractorParams(obj,"barkSpectrum"), ...
                    "erbSpectrum",          getFeatureExtractorParams(obj,"erbSpectrum"), ...
                    "mfcc",                 getFeatureExtractorParams(obj,"mfcc"), ...
                    "gtcc",                 getFeatureExtractorParams(obj,"gtcc"), ...
                    "spectralFlux",         getFeatureExtractorParams(obj,"spectralFlux"), ...
                    "spectralRolloffPoint", getFeatureExtractorParams(obj,"spectralRolloffPoint"), ...
                    "pitch",                getFeatureExtractorParams(obj,"pitch"), ...
                    "zerocrossrate",        getFeatureExtractorParams(obj,"zerocrossrate") ...
                    );
                
                if ~displayAll
                    setIntermediaryPipelineVariables(obj)
                    if ~obj.pCalculateLinearSpectrum
                        params = rmfield(params,'linearSpectrum');
                    end
                    if ~obj.pCalculateMelSpectrum
                        params = rmfield(params,'melSpectrum');
                    end
                    if ~obj.pCalculateBarkSpectrum
                        params = rmfield(params,'barkSpectrum');
                    end
                    if ~obj.pCalculateERBSpectrum
                        params = rmfield(params,'erbSpectrum');
                    end
                    if ~obj.pCalculateMFCC
                        params = rmfield(params,'mfcc');
                    end
                    if ~obj.pCalculateGTCC
                        params = rmfield(params,'gtcc');
                    end
                    if ~obj.pitch
                        params = rmfield(params,'pitch');
                    end
                    if ~obj.spectralFlux
                        params = rmfield(params,'spectralFlux');
                    end
                    if ~obj.spectralRolloffPoint
                        params = rmfield(params,'spectralRolloffPoint');
                    end
                    if ~obj.zerocrossrate
                        params = rmfield(params,'zerocrossrate');
                    end
                end
            end
        end
    end
    
    % Hidden Methods
    methods (Hidden = true)
        y = getFeatureExtractorDefaultParams(obj,~)
    end
    
    % Private Methods
    methods (Access = private)
        y = getFeatureExtractorParams(obj,~)
        y = getNumFeaturePerExtractor(obj,~)
        y = getFeatureExtractUserSpecifiedParams(obj,~)
        [outputIndex,outputIndexReduced,featuresPerHop] = getOutputIndex(obj)
        [allParams,userSpecParams] = parseObsoleteParams(obj,featureName,allParams,userSpecParams)
        function setup(obj,prototype)
            obj.pPrototype = prototype;
            if isa(prototype,'gpuArray')
                coder.internal.errorIf(obj.zerocrossrate,'audio:audioFeatureExtractor:FeatureDoesNotSupportgpuArray','zerocrossrate')
            end

            % Validate FFT length
            coder.internal.errorIf(obj.pFFTLength < numel(obj.Window), ...
                'dsp:system:STFT:FFTLengthTooShort')
            
            % Validate overlap length
            coder.internal.errorIf(obj.OverlapLength >= numel(obj.Window), ...
                'audio:audioFeatureExtractor:OverlapTooLong')
            
            setIntermediaryPipelineVariables(obj)
            obj.pPipelineParameters = [];
            
            ossb = audio.internal.getOnesidedFFTRange(obj.pFFTLength);
            logical_ossb = false(obj.pFFTLength,1);
            logical_ossb(ossb) = true;
            obj.pPipelineParameters.OneSidedSpectrumBins = ossb;
            obj.pPipelineParameters.OneSidedSpectrumBinsLogical = logical_ossb;
            
            if obj.pExtractSpectralDescriptor
                params = getFeatureExtractorParams(obj,obj.SpectralDescriptorInput);
                if strcmpi(obj.SpectralDescriptorInput,"linearSpectrum")
                    range   = params.FrequencyRange;
                    binHigh = floor(range(2)*obj.pFFTLength/obj.SampleRate + 1);
                    binLow  = ceil(range(1)*obj.pFFTLength/obj.SampleRate + 1);
                    bins    = binLow:binHigh;
                    notEnoughBins = numel(bins)==1;
                else % barkSpectrum, melSpectrum, erbSpectrum
                    notEnoughBins = params.NumBands==1;
                end
                coder.internal.errorIf(notEnoughBins, ...
                    'audio:audioFeatureExtractor:InvalidConfigurationSpectrumTooSmall', ...
                    'audioFeatureExtractor',obj.SpectralDescriptorInput)
            end

            coder.internal.errorIf(obj.pCalculateGTCC && getFeatureExtractorParams(obj,'erbSpectrum').NumBands==1, ...
                'audio:audioFeatureExtractor:InvalidConfigurationSpectrumTooSmallCepstrum', ...
                'audioFeatureExtractor','erbSpectrum')

            coder.internal.errorIf(obj.pCalculateMFCC && getFeatureExtractorParams(obj,'melSpectrum').NumBands==1, ...
                'audio:audioFeatureExtractor:InvalidConfigurationSpectrumTooSmallCepstrum', ...
                'audioFeatureExtractor','erbSpectrum')

            % Verify dependent parameter combinations.
            [~,extractorInfo] = info(obj);
            settableExtractors = ["mfcc","gtcc", ...
                "spectralFlux","spectralRolloffPoint", ...
                "melSpectrum","barkSpectrum","linearSpectrum","erbSpectrum", ...
                "pitch", "harmonicRatio"];
            extractors = fieldnames(extractorInfo);
            extractorsToCheck = intersect(settableExtractors,extractors);
            for ii = 1:numel(extractorsToCheck)
                propname = extractorsToCheck(ii);
                userSpecifiedParams = getFeatureExtractUserSpecifiedParams(obj,propname);
                if isempty(userSpecifiedParams)
                    setExtractorParameters(obj,propname)
                else
                    setExtractorParameters(obj,propname,userSpecifiedParams)
                end
            end

            % Save parameters in the extract loop to struct
            obj.pPipelineParameters.Window = cast(obj.Window,'like',prototype);
            obj.pPipelineParameters.WindowLength = numel(obj.Window);
            obj.pPipelineParameters.OverlapLength = double(obj.OverlapLength);
            obj.pPipelineParameters.HopLength = numel(obj.Window) - double(obj.OverlapLength);
            obj.pPipelineParameters.FFTLength = obj.pFFTLength;
            obj.pPipelineParameters.SampleRate = cast(obj.SampleRate,underlyingType(prototype));
            
            if obj.pUseWindowNormalizationPower
                obj.pPipelineParameters.PowerNormalizationFactor = 1/(sum(obj.pPipelineParameters.Window)^2);
            end
            if obj.pUseWindowNormalizationMagnitude
                obj.pPipelineParameters.MagnitudeNormalizationFactor = 1/sum(obj.pPipelineParameters.Window);
            end
            
            if obj.pCalculateLinearSpectrum
                params  = getFeatureExtractorParams(obj,"linearSpectrum");
                range   = params.FrequencyRange;
                binHigh = floor(range(2)*obj.pPipelineParameters.FFTLength/obj.pPipelineParameters.SampleRate + 1);
                binLow  = ceil(range(1)*obj.pPipelineParameters.FFTLength/obj.pPipelineParameters.SampleRate + 1);
                if binLow > binHigh
                    % If the requested frequency range cannot map to a bin range,
                    % choose the single nearest bin.
                    bins = round(mean([range(1)*obj.pPipelineParameters.FFTLength/obj.pPipelineParameters.SampleRate + 1,range(2)*obj.pPipelineParameters.FFTLength/obj.pPipelineParameters.SampleRate + 1]));
                else
                    bins = binLow:binHigh;
                end
                binLogical = false(numel(ossb),1);
                binLogical(bins) = true;
                obj.pPipelineParameters.linearSpectrum.FrequencyBinsLogical = binLogical;
                obj.pPipelineParameters.linearSpectrum.FrequencyBins = bins;
                
                w = (obj.pPipelineParameters.SampleRate/obj.pPipelineParameters.FFTLength)*(bins-1);
                if rem(obj.pFFTLength,2) && binHigh == floor(obj.pFFTLength/2 + 1)
                    w(end) = obj.pPipelineParameters.SampleRate*(obj.pPipelineParameters.FFTLength-1)/(2*obj.pPipelineParameters.FFTLength);
                end
                LinearFc = w(:);
                
                if params.WindowNormalization
                    if strcmpi(params.SpectrumType,'Power')
                        linearNormalizationFactor = obj.pPipelineParameters.PowerNormalizationFactor;
                    else
                        linearNormalizationFactor = obj.pPipelineParameters.MagnitudeNormalizationFactor;
                    end
                    obj.pPipelineParameters.linearSpectrum.NormalizationFactor = cast(2*linearNormalizationFactor,'like',prototype);
                else
                    obj.pPipelineParameters.linearSpectrum.NormalizationFactor = cast(2,'like',prototype);
                end
                if strcmpi(params.SpectrumType,'Power')
                    obj.pPipelineParameters.linearSpectrum.IsPower = true;
                else
                    obj.pPipelineParameters.linearSpectrum.IsPower = false;
                end
                if bins(1)==1 && bins(end) == floor(obj.pPipelineParameters.FFTLength/2+1)
                    obj.pPipelineParameters.linearSpectrum.FullSpectrum = true;
                else
                    obj.pPipelineParameters.linearSpectrum.FullSpectrum = false;
                end
                
                adjustFirstBin = bins(1)==1;
                adjustLastBin = (bins(end) == floor(obj.pPipelineParameters.FFTLength/2+1)) && (rem(obj.pPipelineParameters.FFTLength,2)==0);

                if adjustFirstBin || adjustLastBin
                    adjustment = ones(numel(bins),1);
                    if adjustFirstBin
                        adjustment(1) = 0.5;
                    end
                    if adjustLastBin
                        adjustment(end) = 0.5;
                    end
                    obj.pPipelineParameters.linearSpectrum.AdjustBins = true;
                    obj.pPipelineParameters.linearSpectrum.AdjustBinsVector = cast(adjustment,'like',prototype);
                else
                    obj.pPipelineParameters.linearSpectrum.AdjustBins = false;
                end
            end
            if obj.pCalculateMelSpectrum
                params = getFeatureExtractorParams(obj,"melSpectrum");
                [fb,MelFilterBankFc] = designAuditoryFilterBank(obj.SampleRate, ...
                    "OneSided",true, ...
                    "FrequencyScale","mel", ...
                    "FFTLength",obj.pFFTLength, ...
                    "NumBands",params.NumBands, ...
                    "FrequencyRange",params.FrequencyRange, ...
                    "Normalization",params.FilterBankNormalization,...
                    "FilterBankDesignDomain",params.FilterBankDesignDomain);
                if strcmpi(obj.pmelSpectrumType,'power')
                    if obj.pmelSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.PowerNormalizationFactor;
                    end
                    obj.pPipelineParameters.melSpectrum.IsPower = true;
                else
                    if obj.pmelSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.MagnitudeNormalizationFactor;
                    end
                    obj.pPipelineParameters.melSpectrum.IsPower = false;
                end
                obj.pPipelineParameters.melSpectrum.FilterBank = cast(fb,'like',prototype);
            end
            if obj.pCalculateBarkSpectrum
                params = getFeatureExtractorParams(obj,"barkSpectrum");
                [fb,BarkFilterBankFc] = designAuditoryFilterBank(obj.SampleRate, ...
                    "OneSided",true, ...
                    "FrequencyScale","bark", ...
                    "FFTLength",obj.pFFTLength, ...
                    "FrequencyRange",params.FrequencyRange, ...
                    "NumBands",params.NumBands, ...
                    "Normalization",params.FilterBankNormalization,...
                    "FilterBankDesignDomain",params.FilterBankDesignDomain);
                
                if strcmpi(obj.pbarkSpectrumType,'power')
                    if obj.pbarkSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.PowerNormalizationFactor;
                    end
                    obj.pPipelineParameters.barkSpectrum.IsPower = true;
                else
                    if obj.pbarkSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.MagnitudeNormalizationFactor;
                    end
                    obj.pPipelineParameters.barkSpectrum.IsPower = false;
                end
                obj.pPipelineParameters.barkSpectrum.FilterBank = cast(fb,'like',prototype);
            end
            if obj.pCalculateERBSpectrum
                params = getFeatureExtractorParams(obj,"erbSpectrum");
                [fb,ERBFilterBankFc] = designAuditoryFilterBank(obj.SampleRate, ...
                    "OneSided",true, ...
                    "FrequencyScale","erb", ...
                    "FFTLength",obj.pFFTLength, ...
                    "FrequencyRange",params.FrequencyRange, ...
                    "NumBands",params.NumBands, ...
                    "Normalization",params.FilterBankNormalization);
                
                if strcmpi(obj.perbSpectrumType,'power')
                    if obj.perbSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.PowerNormalizationFactor;
                    end
                    obj.pPipelineParameters.erbSpectrum.IsPower = true;
                else
                    if obj.perbSpectrumWinNorm
                        fb = fb*obj.pPipelineParameters.MagnitudeNormalizationFactor;
                    end
                    obj.pPipelineParameters.erbSpectrum.IsPower = false;
                end
                obj.pPipelineParameters.erbSpectrum.FilterBank = cast(fb,'like',prototype);
            end
            
            if obj.pExtractSpectralDescriptor
                switch obj.SpectralDescriptorInput
                    case "linearSpectrum"
                        obj.pPipelineParameters.SpectralDescriptorInput.FrequencyVector = cast(LinearFc,'like',prototype);
                    case "melSpectrum"
                        obj.pPipelineParameters.SpectralDescriptorInput.FrequencyVector = cast(MelFilterBankFc,'like',prototype);
                    case "barkSpectrum"
                        obj.pPipelineParameters.SpectralDescriptorInput.FrequencyVector = cast(BarkFilterBankFc,'like',prototype);
                    case "erbSpectrum"
                        obj.pPipelineParameters.SpectralDescriptorInput.FrequencyVector = cast(ERBFilterBankFc,'like',prototype);
                end
            end
            
            if obj.pCalculateMFCC
                mfccParams = getFeatureExtractorParams(obj,"mfcc");
                melSpectrumParams = getFeatureExtractorParams(obj,"melSpectrum");
                coder.internal.errorIf(melSpectrumParams.NumBands < mfccParams.NumCoeffs, ...
                    'audio:audioFeatureExtractor:BadNumCoeffs','mfcc','melSpectrum');
                dctMatrix = audio.internal.createDCTmatrix(mfccParams.NumCoeffs, ...
                    melSpectrumParams.NumBands,underlyingType(prototype));
                obj.pPipelineParameters.mfcc.DCTMatrix = cast(dctMatrix,'like',prototype);
                if obj.mfccDelta || obj.mfccDeltaDelta
                    M = floor(mfccParams.DeltaWindowLength/2);
                    b = (M:-1:-M)./sum((1:M).^2);
                    obj.pPipelineParameters.mfcc.deltaFilterCoeffs = cast(b,'like',prototype);
                end
                obj.pPipelineParameters.mfcc.Parameters = mfccParams;
            end
            if obj.pCalculateGTCC
                gtccParams = getFeatureExtractorParams(obj,"gtcc");
                erbSpectrumParams = getFeatureExtractorParams(obj,"erbSpectrum");
                coder.internal.errorIf(erbSpectrumParams.NumBands < gtccParams.NumCoeffs, ...
                    'audio:audioFeatureExtractor:BadNumCoeffs','gtcc','erbSpectrum');
                dctMatrix = audio.internal.createDCTmatrix(gtccParams.NumCoeffs, ...
                    erbSpectrumParams.NumBands,underlyingType(prototype));
                obj.pPipelineParameters.gtcc.DCTMatrix = cast(dctMatrix,'like',prototype);
                if obj.gtccDelta || obj.gtccDeltaDelta
                    M = floor(gtccParams.DeltaWindowLength/2);
                    b = (M:-1:-M)./sum((1:M).^2);
                    obj.pPipelineParameters.gtcc.deltaFilterCoeffs = cast(b,'like',prototype);
                end
                obj.pPipelineParameters.gtcc.Parameters = gtccParams;
            end
            if obj.spectralFlux
                obj.pPipelineParameters.spectralFlux.Parameters = getFeatureExtractorParams(obj,"spectralFlux");
            end
            if obj.spectralRolloffPoint
                obj.pPipelineParameters.spectralRolloffPoint.Parameters = getFeatureExtractorParams(obj,"spectralRolloffPoint");
            end
            if obj.pitch
                obj.pPipelineParameters.pitch.Parameters = getFeatureExtractorParams(obj,"pitch");
            end
            if obj.zerocrossrate
                obj.pPipelineParameters.zerocrossrate.Parameters = getFeatureExtractorParams(obj,"zerocrossrate");
            end
            [obj.pOutputIndex,obj.pOutputIndexReduced,obj.pPipelineParameters.FeaturesPerHop] = getOutputIndex(obj);
            
            obj.pIsInitialized = true;
        end
        function setIntermediaryPipelineVariables(obj)
            obj.pUseMagnitudeSpectrum = false;
            obj.pUsePowerSpectrum = false;
            
            obj.pExtractSpectralDescriptor = obj.spectralCentroid || obj.spectralCrest || ...
                obj.spectralDecrease || obj.spectralEntropy || obj.spectralFlatness || ...
                obj.spectralFlux || obj.spectralKurtosis || obj.spectralRolloffPoint || ...
                obj.spectralSkewness || obj.spectralSlope || obj.spectralSpread;
            
            obj.pUseSpectrum = obj.pExtractSpectralDescriptor || ...
                obj.melSpectrum || obj.barkSpectrum || obj.erbSpectrum || obj.linearSpectrum || ...
                obj.mfcc || obj.mfccDelta || obj.mfccDeltaDelta || ...
                obj.gtcc || obj.gtccDelta || obj.gtccDeltaDelta;
            
            obj.pCalculateGTCC = obj.gtcc || obj.gtccDelta || obj.gtccDeltaDelta;
            obj.pCalculateMFCC = obj.mfcc || obj.mfccDelta || obj.mfccDeltaDelta;
            
            obj.pCalculateBarkSpectrum   = obj.barkSpectrum   || (strcmpi(obj.SpectralDescriptorInput,"barkSpectrum")   && obj.pExtractSpectralDescriptor);
            obj.pCalculateMelSpectrum    = obj.melSpectrum    || (strcmpi(obj.SpectralDescriptorInput,"melSpectrum")    && obj.pExtractSpectralDescriptor) || obj.pCalculateMFCC;
            obj.pCalculateLinearSpectrum = obj.linearSpectrum || (strcmpi(obj.SpectralDescriptorInput,"linearSpectrum") && obj.pExtractSpectralDescriptor);
            obj.pCalculateERBSpectrum    = obj.erbSpectrum    || (strcmpi(obj.SpectralDescriptorInput,"erbSpectrum")    && obj.pExtractSpectralDescriptor) || obj.pCalculateGTCC;
            
            obj.pUseWindowNormalizationMagnitude = false;
            obj.pUseWindowNormalizationPower = false;
            if obj.pCalculateLinearSpectrum
                params = getFeatureExtractorParams(obj,'linearSpectrum');
                if strcmpi(obj.plinearSpectrumType,"Magnitude")
                    obj.pUseMagnitudeSpectrum = true;
                    obj.pUseWindowNormalizationMagnitude = obj.pUseWindowNormalizationMagnitude | params.WindowNormalization;
                else
                    obj.pUsePowerSpectrum = true;
                    obj.pUseWindowNormalizationPower = obj.pUseWindowNormalizationPower | params.WindowNormalization;
                end
            end
            if obj.pCalculateMelSpectrum
                params = getFeatureExtractorParams(obj,'melSpectrum');
                if strcmpi(obj.pmelSpectrumType,"Magnitude")
                    obj.pUseMagnitudeSpectrum = true;
                    obj.pUseWindowNormalizationMagnitude = obj.pUseWindowNormalizationMagnitude | params.WindowNormalization;
                else
                    obj.pUsePowerSpectrum = true;
                    obj.pUseWindowNormalizationPower = obj.pUseWindowNormalizationPower | params.WindowNormalization;
                end
            end
            if obj.pCalculateBarkSpectrum
                params = getFeatureExtractorParams(obj,'barkSpectrum');
                if strcmpi(obj.pbarkSpectrumType,"Magnitude")
                    obj.pUseMagnitudeSpectrum = true;
                    obj.pUseWindowNormalizationMagnitude = obj.pUseWindowNormalizationMagnitude | params.WindowNormalization;
                else
                    obj.pUsePowerSpectrum = true;
                    obj.pUseWindowNormalizationPower = obj.pUseWindowNormalizationPower | params.WindowNormalization;
                end
            end
            if obj.pCalculateERBSpectrum
                params = getFeatureExtractorParams(obj,'erbSpectrum');
                if strcmpi(obj.perbSpectrumType,"Magnitude")
                    obj.pUseMagnitudeSpectrum = true;
                    obj.pUseWindowNormalizationMagnitude = obj.pUseWindowNormalizationMagnitude | params.WindowNormalization;
                else
                    obj.pUsePowerSpectrum = true;
                    obj.pUseWindowNormalizationPower = obj.pUseWindowNormalizationPower | params.WindowNormalization;
                end
            end
            if obj.shortTimeEnergy && obj.pUseSpectrum
                obj.pUsePowerSpectrum = true;
            end
        end
    end
    
    % Protected Methods - Display
    methods (Access = protected)
        function groups = getPropertyGroups(~)
            mainProps = { ...
                'Window', ...
                'OverlapLength', ...
                'SampleRate', ...
                'FFTLength', ...
                'SpectralDescriptorInput', ...
                'FeatureVectorLength'};
            groups = matlab.mixin.util.PropertyGroup(mainProps,'Properties');
        end
        function footer = getFooter(obj)
            featureNames = string(fieldnames(obj.pFeaturesToExtract));
            isEnabled    = structfun(@(x)(x),obj.pFeaturesToExtract);
            
            featureNamesEnabled = featureNames(isEnabled);
            featureNamesDisabled = featureNames(~isEnabled);
            
            FPL = 6; % Features displayed per line
            if ~isempty(featureNamesEnabled)
                numToPad = FPL*ceil(numel(featureNamesEnabled)/FPL) - numel(featureNamesEnabled);
                featureNamesEnabled = [featureNamesEnabled;strings(numToPad,1)];
                featureNamesEnabled = reshape(featureNamesEnabled,FPL,[])';
                temp = cell(size(featureNamesEnabled,1),1);
                for i = 1:size(featureNamesEnabled,1)
                    if i == size(featureNamesEnabled,1)
                        a = strjoin(featureNamesEnabled(i,1:end-numToPad),', ');
                    else
                        a = strjoin(featureNamesEnabled(i,:),', ');
                    end
                    b = ["    ",a];
                    c = strjoin(b);
                    temp{i} = c;
                end
                featureNamesEnabled = strjoin([temp{:}],'\n');
            else
                featureNamesEnabled = "     none";
            end
            
            if ~isempty(featureNamesDisabled)
                numToPad = FPL*ceil(numel(featureNamesDisabled)/FPL) - numel(featureNamesDisabled);
                featureNamesDisabled = [featureNamesDisabled;strings(numToPad,1)];
                featureNamesDisabled = reshape(featureNamesDisabled,FPL,[])';
                temp = cell(size(featureNamesDisabled,1),1);
                for i = 1:size(featureNamesDisabled,1)
                    if i == size(featureNamesDisabled,1)
                        a = strjoin(featureNamesDisabled(i,1:end-numToPad),', ');
                    else
                        a = strjoin(featureNamesDisabled(i,:),', ');
                    end
                    b = ["    ",a];
                    c = strjoin(b);
                    temp{i} = c;
                end
                featureNamesDisabled = strjoin(cat(1,temp{:}),'\n');
            else
                featureNamesDisabled = "     none";
            end
            
            formatSpec = ['   ',getString(message('audio:audioFeatureExtractor:EnabledFeatures')),'\n%s\n\n   ', ...
                getString(message('audio:audioFeatureExtractor:DisabledFeatures')),'\n%s\n\n\n   ', ...
                getString(message('audio:audioFeatureExtractor:DisplayExampleIntro')),'\n   ', ...
                getString(message('audio:audioFeatureExtractor:DisplayExample')),'\n'];
            text = sprintf(formatSpec,featureNamesEnabled,featureNamesDisabled);
            
            footer = text;
        end
    end
    
    % Hidden Methods
    methods (Hidden)
        function setExtractorParams(obj,propname,varargin)
            setExtractorParameters(obj,propname,varargin{:})
        end
    end

    % Static Methods - Utilities
    methods (Hidden, Static)
        y = mergeStructs(obj,~,~)
        y = reduceStruct(obj,~)
    end
    
    % Public Methods - Sets/Gets
    methods
        function set.linearSpectrum(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.linearSpectrum','linearSpectrum')
            obj.linearSpectrum = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.melSpectrum(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.melSpectrum','melSpectrum')
            obj.melSpectrum = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.barkSpectrum(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.barkSpectrum','barkSpectrum')
            obj.barkSpectrum = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.erbSpectrum(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.erbSpectrum','erbSpectrum')
            obj.erbSpectrum = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.mfcc(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.mfcc','mfcc')
            obj.mfcc = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.mfccDelta(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.mfccDelta','mfccDelta')
            obj.mfccDelta = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.mfccDeltaDelta(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.mfccDeltaDelta','mfccDeltaDelta')
            obj.mfccDeltaDelta = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.gtcc(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.gtcc','gtcc')
            obj.gtcc = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.gtccDelta(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.gtccDelta','gtccDelta')
            obj.gtccDelta = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.gtccDeltaDelta(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.gtccDeltaDelta','gtccDeltaDelta')
            obj.gtccDeltaDelta = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralCentroid(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralCentroid','spectralCentroid')
            obj.spectralCentroid = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralCrest(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralCrest','spectralCrest')
            obj.spectralCrest = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralDecrease(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralDecrease','spectralDecrease')
            obj.spectralDecrease = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralEntropy(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralEntropy','spectralEntropy')
            obj.spectralEntropy = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralFlatness(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralFlatness','spectralFlatness')
            obj.spectralFlatness = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralFlux(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralFlux','spectralFlux')
            obj.spectralFlux = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralKurtosis(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralKurtosis','spectralKurtosis')
            obj.spectralKurtosis = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralRolloffPoint(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralRolloffPoint','spectralRolloffPoint')
            obj.spectralRolloffPoint = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralSlope(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralSlope','spectralSlope')
            obj.spectralSlope = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralSpread(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralSpread','spectralSpread')
            obj.spectralSpread = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.spectralSkewness(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.spectralSkewness','spectralSkewness')
            obj.spectralSkewness = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.pitch(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.pitch','pitch')
            obj.pitch = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.harmonicRatio(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.harmonicRatio','harmonicRatio')
            obj.harmonicRatio = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.zerocrossrate(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.zerocrossrate','zerocrossrate')
            obj.zerocrossrate = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.shortTimeEnergy(obj,val)
            validateattributes(val,{'logical','numeric'}, ...
                {'scalar','nonnan','real'},'set.shortTimeEnergy','shortTimeEnergy')
            obj.shortTimeEnergy = logical(gather(val));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.Window(obj,val)
            validateattributes(val,{'single','double'}, ...
                {'vector','real','finite'},'set.Window','Window')
            obj.Window = gather(val(:));
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.OverlapLength(obj,val)
            validateattributes(val,{'single','double'}, ...
                {'nonnegative','scalar','integer'},'set.OverlapLength','OverlapLength')
            obj.OverlapLength = gather(val);
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.SampleRate(obj,val)
            validateattributes(val,{'single','double'}, ...
                {'positive','scalar','real','finite'},'set.SampleRate','SampleRate')
            obj.SampleRate = gather(val);
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.FFTLength(obj,val)
            if isempty(val)
                validateattributes(val, {'numeric'},{}, ...
                    'set.FFTLength','FFTLength');
            else
                validateattributes(val, {'numeric'}, ...
                    {'finite','real','scalar','integer','positive'}, ...
                    'set.FFTLength','FFTLength');
            end
            obj.FFTLength = gather(val);
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function set.SpectralDescriptorInput(obj,val)
            obj.SpectralDescriptorInput = gather(val);
            obj.pIsInitialized = false; %#ok<MCSUP>
        end
        function out = get.pFeaturesToExtract(obj)
            out = struct( ...
                'linearSpectrum',      obj.linearSpectrum, ...
                'melSpectrum',         obj.melSpectrum, ...
                'barkSpectrum',        obj.barkSpectrum, ...
                'erbSpectrum',         obj.erbSpectrum, ...
                'mfcc',                obj.mfcc, ...
                'mfccDelta',           obj.mfccDelta, ...
                'mfccDeltaDelta',      obj.mfccDeltaDelta, ...
                'gtcc',                obj.gtcc, ...
                'gtccDelta',           obj.gtccDelta, ...
                'gtccDeltaDelta',      obj.gtccDeltaDelta, ...
                'spectralCentroid',    obj.spectralCentroid, ...
                'spectralCrest',       obj.spectralCrest, ...
                'spectralDecrease',    obj.spectralDecrease, ...
                'spectralEntropy',     obj.spectralEntropy, ...
                'spectralFlatness',    obj.spectralFlatness, ...
                'spectralFlux',        obj.spectralFlux, ...
                'spectralKurtosis',    obj.spectralKurtosis, ...
                'spectralRolloffPoint',obj.spectralRolloffPoint, ...
                'spectralSkewness',    obj.spectralSkewness, ...
                'spectralSlope',       obj.spectralSlope, ...
                'spectralSpread',      obj.spectralSpread, ...
                'pitch',               obj.pitch, ...
                'harmonicRatio',       obj.harmonicRatio, ...
                "zerocrossrate",       obj.zerocrossrate, ...
                "shortTimeEnergy",     obj.shortTimeEnergy);
        end
        function out = get.pFFTLength(obj)
            if isempty(obj.FFTLength)
                out =  numel(obj.Window);
            else
                out =  obj.FFTLength;
            end
        end
        function out = get.FeatureVectorLength(obj)
            [~,~,out] = getOutputIndex(obj);
        end
    end
    
end