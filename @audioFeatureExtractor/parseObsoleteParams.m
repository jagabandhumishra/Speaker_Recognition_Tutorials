function [allParams,userSpecParams] = parseObsoleteParams(~,featureName,allParams,userSpecParams)
%parseObsoleteParams Parse obsolete extractor parameters
%
%   This function is called in setExtractorParameter method to verify and
%   handle obsolete parameters.

%   Copyright 2020 The MathWorks, Inc.

% Normalization has been renamed to FilterBankNormalization.
if any(strcmp(featureName,{'melSpectrum','barkSpectrum','erbSpectrum'}))
    % copy Normalization value to FilterBankNormalization
    if isfield(userSpecParams,'Normalization')
        % Both Normalization and FilterBankNormalization cannot be
        % specified to conflicting values
        val = userSpecParams.Normalization;
        assert(~isfield(userSpecParams,'FilterBankNormalization') || ...
            isequal(userSpecParams.FilterBankNormalization,val), ...
            message('audio:audioFeatureExtractor:ConflictingNormalizationValue', ...
            'Normalization','FilterBankNormalization'));
        
        % copy the value to FilterBankNormalization
        allParams.FilterBankNormalization = val;
        userSpecParams.FilterBankNormalization = val;
        userSpecParams = rmfield(userSpecParams,'Normalization');
    end
    
    % remove Normalization from allParams
    allParams = rmfield(allParams,'Normalization');
end
