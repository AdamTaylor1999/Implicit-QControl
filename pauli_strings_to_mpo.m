function operator_strings = pauli_strings_to_mpo(n, pauli_strings)
%   Converts a list of Pauli strings into a list of MPOs.
%   Takes a cell array of Pauli strings (e.g., {'XIYIZ','ZZIII'}) and
%   the number of sites n, and returns a cell array of MPOs.

    % Define Pauli matrices
    id = eye(2);
    sx = [0 1; 1 0];
    sy = [0 -1i; 1i 0];
    sz = [1 0; 0 -1];

    % Create dictionary of Pauli matrices. containers. Map used as
    % dictionary
    pauli_dict = containers.Map({'I', 'X', 'Y', 'Z'}, {id, sx, sy, sz});

    num_strings = length(pauli_strings);

    % Construct MPOs
    operator_strings = cell(num_strings, 1);
    for k = 1:num_strings
        pauli_str = pauli_strings{k};
        hcell = cell(1, n);
        for j = 1:n
            op = pauli_dict(pauli_str(j));
            hcell{j} = reshape(op, [1, 2, 2, 1]);  % MPO format: [left, physical_in, physical_out, right]
        end
        operator_strings{k} = hcell_to_mpo(hcell); 
    end
end
