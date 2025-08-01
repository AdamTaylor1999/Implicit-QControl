function pauli_strings = local_2body_strings(n, local_2body_int)
%Returns a cell of all local 2body strings with the local_2body_interaction

%eg; local_2body_strings(5, 'XY') = {'XYIII', 'IXYII', 'IIXYI', 'IIIXY'}

pauli_strings = cell(1, n - 1);
for j = 1:n - 1
    pauli_strings{j} = pauli_string_at_qubit_position(n, local_2body_int, j);
end

