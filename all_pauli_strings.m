function L = all_pauli_strings(n)
%Returns all Pauli strings of an n qubit system as a cell\

L = {'I', 'X', 'Y', 'Z'};
while length(L) < 4^n
    for j = 1:length(L)
        L = [L, {[L{j}, 'X'], [L{j}, 'Y'], [L{j}, 'Z']}];
        L{j} = [L{j}, 'I'];
    end
end
