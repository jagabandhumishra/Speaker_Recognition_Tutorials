function featureExtractorParams = getFeatureExtractorParams(obj,featurename)
% featureExtractorParams = getFeatureExtractorParams(obj,featurename)
% returns the parameters of the specified feature name. The parameters
% returned are the union of the user-specified parameters and the default
% parameters.

% Copyright 2020 The MathWorks, Inc.
defaults = getFeatureExtractorDefaultParams(obj,featurename);
userSpecifiedParams = getFeatureExtractUserSpecifiedParams(obj,featurename);

if strcmpi(featurename,'erbSpectrum')
    % erbSpectrum has interdependent properties. If the user
    % specified FrequencyRange but not NumBands, change the default
    % of NumBands.
    if isfield(userSpecifiedParams,"FrequencyRange") && ...
            ~isfield(userSpecifiedParams,"NumBands")
        FR = userSpecifiedParams.FrequencyRange;
        defaults.NumBands = ceil(hz2erb(FR(2)) - hz2erb(FR(1)));
    end
end

featureExtractorParams = audioFeatureExtractor.mergeStructs(defaults,userSpecifiedParams);
end