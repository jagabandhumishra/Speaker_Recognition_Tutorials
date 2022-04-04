function aStruct = reduceStruct(aStruct)
% aStruct = reduceStruct(aStruct) removes any field-value pairs in aStruct
% if the value is [].

% Copyright 2020 The MathWorks, Inc.
fn = fieldnames(aStruct);
for i = 1:numel(fn)
    if isempty(aStruct.(fn{i}))
        aStruct = rmfield(aStruct,fn{i});
    end
end
end