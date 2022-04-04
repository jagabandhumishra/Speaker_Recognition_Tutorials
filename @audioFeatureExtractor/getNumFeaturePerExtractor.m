function numFeaturePerExtractor = getNumFeaturePerExtractor(obj)
% numFeaturePerExtractor = getNumFeaturePerExtractor(obj) returns a struct
% with fields corresponding to the features to extract, and values
% corresponding to the number of elements in each feature.

% Copyright 2020-2021 The MathWorks, Inc.

numFeaturePerExtractor = obj.pFeaturesToExtract;
if obj.linearSpectrum
    params  = getFeatureExtractorParams(obj,"linearSpectrum");
    range   = params.FrequencyRange;
    binHigh = floor(range(2)*obj.pFFTLength/obj.SampleRate + 1);
    binLow  = ceil(range(1)*obj.pFFTLength/obj.SampleRate + 1);
    if binLow > binHigh
        %If the requested frequency range cannot map to a bin range,
        %choose the single nearest bin.
        bins = round(mean([range(1)*obj.pFFTLength/obj.SampleRate + 1,range(2)*obj.pFFTLength/obj.SampleRate + 1]));
    else
        bins = binLow:binHigh;
    end
    numFeaturePerExtractor.linearSpectrum = numel(bins);
end
if obj.melSpectrum
    params = getFeatureExtractorParams(obj,"melSpectrum");
    numFeaturePerExtractor.melSpectrum = params.NumBands;
end
if obj.barkSpectrum
    params = getFeatureExtractorParams(obj,"barkSpectrum");
    numFeaturePerExtractor.barkSpectrum = params.NumBands;
end
if obj.erbSpectrum
    params = getFeatureExtractorParams(obj,"erbSpectrum");
    numFeaturePerExtractor.erbSpectrum = params.NumBands;
end
if obj.mfcc
    params = getFeatureExtractorParams(obj,"mfcc");
    numFeaturePerExtractor.mfcc = params.NumCoeffs;
end
if obj.mfccDelta
    params = getFeatureExtractorParams(obj,"mfcc");
    numFeaturePerExtractor.mfccDelta = params.NumCoeffs;
end
if obj.mfccDeltaDelta
    params = getFeatureExtractorParams(obj,"mfcc");
    numFeaturePerExtractor.mfccDeltaDelta = params.NumCoeffs;
end
if obj.gtcc
    params = getFeatureExtractorParams(obj,"gtcc");
    numFeaturePerExtractor.gtcc = params.NumCoeffs;
end
if obj.gtccDelta
    params = getFeatureExtractorParams(obj,"gtcc");
    numFeaturePerExtractor.gtccDelta = params.NumCoeffs;
end
if obj.gtccDeltaDelta
    params = getFeatureExtractorParams(obj,"gtcc");
    numFeaturePerExtractor.gtccDeltaDelta = params.NumCoeffs;
end
h = {'spectralKurtosis','spectralSkewness','spectralSkewness','spectralSpread', ...
    'spectralCentroid','spectralCrest','spectralDecrease','spectralEntropy', ...
    'spectralFlatness','spectralFlux','spectralRolloffPoint','spectralSlope', ...
    'pitch','harmonicRatio','zerocrossrate','shortTimeEnergy'};
for ii = 1:numel(h)
    numFeaturePerExtractor.(h{ii}) = double(obj.(h{ii}));
end
end