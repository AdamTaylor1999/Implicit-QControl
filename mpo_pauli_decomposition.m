function weights = mpo_pauli_decomposition(n, pauli_basis, mpo)
% Decomposes an MPO over a given Pauli basis. If length(pauli_basis) != 4^n
% then the MPO will not be fully characterised.

%This needs to be fixed to include rescaling the weights! This will likely
%require knowing the norm of the physical operator that the MPO represents.

m = length(pauli_basis);
weights = cell(m, 1);
for j=1:m
    weights{j} = mpo_overlap(pauli_strings_to_mpo(n, {pauli_basis{j}}));
end
