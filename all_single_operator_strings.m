function pauli_strings = all_single_operator_strings(n, operator)
%Returns all Pauli strings consisting of exclusively operator and the
%identity

%eg; all_single_operator_string(3, 'Z') = {'ZII', 'IZI', 'IIZ'}

pauli_strings = cell(1, n);
for j = 1:n
    pauli_strings{j} = pauli_string_at_qubit_position(n, operator, j);
end