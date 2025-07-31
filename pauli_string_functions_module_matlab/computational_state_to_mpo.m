function computational_state_mpo = computational_state_to_mpo(comp_state)
%   Take a list, comp_state, representing a computational state and 
%   converts it into an MPO. 
%   eg; computational_state_to_mpo([0 1 0 1 1 0]) = MPO( |010110><010110| )

n = length(comp_state);

q0 = [1 0; 0 0];
q1 = [0 0; 0 1];

comp_dict = containers.Map({0, 1}, {q0, q1});

hcell = cell(1, n);
for j = 1:n
    op = comp_dict(comp_state(j));
    hcell{j} = reshape(op, [1, 2, 2, 1]);
end
computational_state_mpo = hcell_to_mpo(hcell);

end