function operator_strings = pauli_strings_to_mpo(n, pauli_strings, weights)
%   Converts a list of Pauli strings into a list of MPOs.
%   If no weights are passed, take a cell array of Pauli strings 
%   (e.g., {'XIYIZ','ZZIII'}) and the number of sites n, and returns a cell
%   array of MPOs.

%   If weights are passed, then this weights each Pauli string. Needs to be
%   a matrix of shape [1, length(pauli_strings)]
%   eg; (4, {'IXYZ', 'ZZZZ'}, [0.4, 1.5] gf) --> {MPO(0.4 IXYZ), MPO(1.5
%   ZZZZ)}


    if nargin < 3 || isempty(weights)
        weights = ones(1, length(pauli_strings));
    end

    % Define Pauli matrices
    id = eye(2);
    sx = [0 1; 1 0];
    sy = [0 -1i; 1i 0];
    sz = [1 0; 0 -1];

    % Create dictionary of Pauli matrices. containers. Map used as
    % dictionary
    pauli_dict = containers.Map({'I', 'X', 'Y', 'Z'}, {id, sx, sy, sz});

    num_strings = length(pauli_strings);

    % Construct MPOs -- is this wrong at the edge?
    operator_strings = cell(num_strings, 1);
    for k = 1:num_strings
        pauli_str = pauli_strings{k};
        hcell = cell(1, n);
        for j = 1:n
            op = weights(k) * pauli_dict(pauli_str(j));  %weighted the operator
            hcell{j} = reshape(op, [1, 2, 2, 1]); % MPO format: [left, physical_in, physical_out, right]
        end
        operator_strings{k} = hcell_to_mpo(hcell); 
    end
end