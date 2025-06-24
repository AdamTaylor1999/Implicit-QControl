function labelled_weights = mpo_pauli_decomposition(n, pauli_strings, mpo)
% Decomposes an MPO over a given Pauli basis. If length(pauli_basis) != 4^n
% then the MPO will not be fully characterised.

%This needs to be fixed to include rescaling the weights! This will likely
%require knowing the norm of the physical operator that the MPO represents.

m = length(pauli_strings);
pauli_basis = pauli_strings_to_mpo(n, pauli_strings);

weights = cell(1, m); %calculates the weights of each Pauli string
for j = 1:m
    weights{j} = mpo_overlap(pauli_basis{j}, mpo);
end

magnitudes = cell(1, m); %finds the magnitude in order to order
for j = 1:m
    magnitudes{j} = abs(weights{j});
end
[~, perm] = sort(cell2mat(magnitudes), 'descend');

labelled_weights = dictionary(); %labels the ordered weights
for j = 1:m
    labelled_weights(pauli_strings{perm(j)}) = weights{perm(j)};
end

