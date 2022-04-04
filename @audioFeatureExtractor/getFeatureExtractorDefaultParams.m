function defaults = getFeatureExtractorDefaultParams(obj,featurename)
%getFeatureExtractorDefaultParams Get default extractor parameter values

%   Copyright 2019-2021 The MathWorks, Inc.
fs = obj.SampleRate;
switch featurename
    case 'mfcc'
        defaults = struct("NumCoeffs",13,"DeltaWindowLength",9,"Rectification","log");
    case 'gtcc'
        defaults = struct("NumCoeffs",13,"DeltaWindowLength",9,"Rectification","log");
    case 'spectralFlux'
        defaults = struct("NormType",2);
    case 'spectralRolloffPoint'
        defaults = struct("Threshold",0.95);
    case 'linearSpectrum'
        defaults = struct("FrequencyRange",[0,fs/2],"SpectrumType","power", ...
            "WindowNormalization",true);
    case 'melSpectrum'
        % Normalization will be removed in a future release. Use
        % FilterBankNormalization instead.
        defaults = struct("NumBands",32,"FrequencyRange",[0,fs/2], ...
            "FilterBankNormalization","bandwidth","FilterBankDesignDomain","linear", ...
            "WindowNormalization",true,"SpectrumType","power");
    case 'barkSpectrum'
        % Normalization will be removed in a future release. Use
        % FilterBankNormalization instead.
        defaults = struct("NumBands",32,"FrequencyRange",[0,fs/2], ...
            "FilterBankNormalization","bandwidth","FilterBankDesignDomain","linear", ...
            "WindowNormalization",true,"SpectrumType","power");
    case 'erbSpectrum'
        % Normalization will be removed in a future release. Use
        % FilterBankNormalization instead.
        FR = [0,fs/2];
        defaults = struct("NumBands",ceil(hz2erb(FR(2)) - hz2erb(FR(1))), ...
            "FrequencyRange",FR,"FilterBankNormalization","bandwidth", ...
            "WindowNormalization",true,"SpectrumType","power");
    case 'pitch'
        defaults = struct("Method","NCF","Range",[50,400],"MedianFilterLength",1);
    case 'zerocrossrate'
        defaults = struct('Method',"difference",'Level',0,'Threshold',0, ...
            'TransitionEdge',"both",'ZeroPositive',false);
end
