
% Quantum control for realising propagators
%
% In this code, we use quantum invariants to find quantum control
% solutions for Hamiltonian simulation. 

% In particular, we focus on trying to simulate Heisenberg dynamics
% governed by
% H_h = \sum_j XjXj+1 + YjYj+1 + ZjZj+1 
% using a control Hamiltonian given in terms of Ising interactions
% {Xj, ZkZk+1}.



addpath("TiQC-ToQC/")


%definitions and bond dimension/compression 
q0 = [1; 0];
q1 = [0; 1];
id = eye(2);
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];
tebd_options=tebd_default_options;
tebd_options.bond_dim = 30;
tebd_options.bond_comp = 10;

%no. of qubits
n = 5;

%initial invariants: initially just considering single Pauli strings, not
%sums or combinations
I0_strings = {'ZII', 'IZI', 'IIZ', 'XXI', 'IXX'};
m = length(I0_strings);
I0_mpos = pauli_strings_to_mpo(3, I0_strings);


%Heisenberg Hamiltonian: H_H = XjXj+1 + YjYj+1 + ZjZj+1
H2q_heis = cell(3 * (n - 1), 1);
for j = 1:n - 1
    H2q_heis{j, 1} = kron(sx, sx);
    H2q_heis{j + (n - 1), 1} = kron(sy, sy);
    H2q_heis{j + 2 * (n - 1), 1} = kron(sz, sz);
end

%time evolution parameters
dt = 0.1;
T_evol = 1;
time_steps = round(T_evol / dt);
times = linspace(0, T_evol, time_steps);

%two qubit Hamiltonian's; what form should this have?
c2q = ones(1, time_steps);

%single qubit Hamiltonian - set to zero without breaking code
H1q = struct('sys', cell(1, 1), 'op', cell(1,1));
H1q.sys = (1:n);
H1q.op = cell(n, 1);
for j = 1:n
    H1q.op{j} = id;
end
c1q = zeros(time_steps, 1);

%time evolution for Ij0 --> IjT
IT_mpos = cell(1, m);
for j = 1:m
    [IT_mpos{j}, d_list] = mpo_evol(H2q_heis, H1q, c2q, c1q, T_evol, I0_mpos{j}, tebd_options);
end








