function pauli_string = pauli_string_at_qubit_position(n, op_str, position)
%Returns a Pauli string of op_str at position position.
%eg; pauli_string_at_qubit_position(5, 'XY', 2) = 'IXYII'
%eg; pauli_string_at_qubit_position(5, 'XY', 4) = 'IIIXY'

%Currently no error messages setup - this needs to be done!

if length(op_str) == n
    pauli_string = op_str;
else
    pauli_string = [repmat('I', 1, (position - 1)), op_str, repmat('I', 1, (n - length(op_str) - position + 1))];
end
