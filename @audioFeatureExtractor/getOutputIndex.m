function [outputIndex,outputIndexReduced,featuresPerHop] = getOutputIndex(obj)
% [outputIndex,outputIndexReduced,featuresPerHop] = getOutputIndex(obj)
% returns the following:
%
%   outputIndex - struct with fields corresponding to all possible
%   features, and values corresponding to the column index of the features
%   output from an audioFeatureExtractor object.
%
%   outputIndexReduced - the same as outputIndex, except that features that
%   are not extracted (i.e. the column index is []), are removed.
%
%   featuresPerHop - the total number of features returned by an
%   audioFeatureExtractor object per hop.

% Copyright 2020 The MathWorks, Inc.
numFeaturePerExtractor = getNumFeaturePerExtractor(obj);
fte = obj.pFeaturesToExtract;
idx = 1;
fn = fieldnames(fte);
numFeatureExtractors = numel(fn);
outputIndex = fte;

for i = 1:numFeatureExtractors
    outputIndex.(fn{i}) = idx:(idx+numFeaturePerExtractor.(fn{i})-1);
    idx = idx + numFeaturePerExtractor.(fn{i});
end

outputIndexReduced = audioFeatureExtractor.reduceStruct(outputIndex);

totalFeatures = struct2cell(numFeaturePerExtractor);
featuresPerHop = sum([totalFeatures{:}],'all');
end