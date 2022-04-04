function userSpecifiedParams = getFeatureExtractUserSpecifiedParams(obj,featurename)
% userSpecifiedParams =
% getFeatureExtractUserSpecifiedParams(obj,featurename) returns the
% user-specified parameters of the specified feature name.

% Copyright 2020-2021 The MathWorks, Inc.

switch featurename
    case 'linearSpectrum'
        userSpecifiedParams = obj.plinearSpectrumUserSpecifiedParams;
    case 'melSpectrum'
        userSpecifiedParams = obj.pmelSpectrumUserSpecifiedParams;
    case 'barkSpectrum'
        userSpecifiedParams = obj.pbarkSpectrumUserSpecifiedParams;
    case 'erbSpectrum'
        userSpecifiedParams = obj.perbSpectrumUserSpecifiedParams;
    case 'mfcc'
        userSpecifiedParams = obj.pmfccUserSpecifiedParams;
    case 'gtcc'
        userSpecifiedParams = obj.pgtccUserSpecifiedParams;
    case 'spectralRolloffPoint'
        userSpecifiedParams = obj.pspectralRolloffPointUserSpecifiedParams;
    case 'spectralFlux'
        userSpecifiedParams = obj.pspectralFluxUserSpecifiedParams;
    case 'pitch'
        userSpecifiedParams = obj.ppitchUserSpecifiedParams;
    case 'zerocrossrate'
        userSpecifiedParams = obj.pzerocrossrateUserSpecifiedParams;
end
end