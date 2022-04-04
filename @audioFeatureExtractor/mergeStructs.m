function C = mergeStructs(A,B)
% C = mergeStructs(A,B) replaces the value of any field-value pair in the
% struct A with the corresponding value of the field-value pair in the
% struct B. If struct B includes field-value pairs not in A, they are
% included in C.

% Copyright 2020 The MathWorks, Inc.

% Set C to the original struct, A.
if isempty(A)
    C = B; % No 'original' struct.
else
    C = A;
    
    % Replace any field-value pairs in C to the value in B.
    fn = fieldnames(A);
    for i = 1:numel(fn)
        if isfield(B,fn{i})
            C.(fn{i}) = B.(fn{i});
        end
    end
    
    % Add any field-value pairs that are in B but not A to C.
    if isstruct(B)
        fn = fieldnames(B);
        for ii = 1:numel(fn)
            C.(fn{ii}) = B.(fn{ii});
        end
    end
end
end