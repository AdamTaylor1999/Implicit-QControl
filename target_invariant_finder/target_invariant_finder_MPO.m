%% target_invariant_finder

% In Ref.[1], the authors used quantum invariants to find optimal quanutm 
% control solutions. This is possible if the control Hamiltonian and target 
% Hamiltonian admit a description in terms of algebras that scale 
% polynomially with respect to the number of qubits. Thusfar, one of the
% key limitations of this approach comes from finding the target invariant.
% 
% In this work, we aim to use matrix product operator (MPO) simulations to
% calculate the target invariant. For particular setups, if the associated 
% algebra is polynomially scaling then we can take the calculated target
% invariant and directly use the method described in Ref.[1]. If the
% algebra is exponentially scaling, then we can try carrying out the
% quantum control optimisation directly on the level of MPOs.
%
% To do this, we need to adiabatically evolve an initial invariant, I0, 
% towards the target invariant, IT, with the Hamiltonian
% H(t) = (1 - f(t)) I0 + f(t) HT + f(t) (1 - f(t)) HP
% where HT is the target Hamiltonian and HP is the a perturbing Hamiltonian
% introduced to prevent energy level crossings. 



% [1] "Quantum control without quantum states", PRX Quantum 5, 00346 (2024)


%% definitions and TEBD options

addpath('/Users/at4018/Downloads/TiQC-ToQC-main/TiQC/mpo_invariant/')
addpath('/Users/at4018/Downloads/TiQC-ToQC-main/ToQC/mps/')

%local dimension
d = 2;

%qubit states
q0 = [1; 0];
q1 = [0; 1];
function q = qubit_basis(b)
    if b == 0 || b == '0'
        q = [1; 0];
    elseif b == 1 || b == '1'
        q = [0; 1];
    else
        error('ValueError: in qubit_basis(b), b is not in {0,1}!')
    end
end

%qubit Pauli operators
id = [1,0; 0,1];
sx = [0,1; 1,0];
sy = [0,-1i; 1i,0];
sz = [1,0; 0,-1];

%TEBD options
tebd_options = tebd_default_options; %see documentation for details
tebd_options.bon_dim = 30; %bond dimension
tebd_options.bond_comp = 10;
Dc = tebd_options.bond_comp; %dimension compressed (bond compression)
sv_min = tebd_options.sv_min; %singular value minimum, 10^{-10}


%% parameters, initial and target states

n = 5;

I0_frequencies = [1, 0, 0, 0, 0];
I0_couplings = [1, 1, 1, 1];

HT_frequencies = [0.6, 1.1, 0.4, 1.9, 1.6];
HT_couplings = dictionary('12',0.8, '23',1.5, '34',0.7, '45',0.3);

% initial and target states - calculated elsewhere with Python
psi0 = '10101';
psiT = '00111'; %Hamming distance of 2, suitably different to psi0

I0_GS_energy_python = -5;
%IT_GS_energy_python = -5;  

mps0 = cell(1, n);
mpsT = cell(1, n);
for j = 1:n
    mps0{j} = reshape(qubit_basis(psi0(j)), [1, d, 1]);
    mpsT{j} = reshape(qubit_basis(psiT(j)), [1, d, 1]);
end

%% initial invariant - construct manually for now, but want to automate!
%I0 = Z1 + Z1Z2 + Z2Z3 + Z3Z4 + Z4Z5
m = 0;
for j = 1:n
    if I0_frequencies(j) ~= 0
        m = m + 1;
    end
    if j < n & I0_couplings(j) ~= 0
        m = m + 1;
    end
end
normI0 = sqrt(m * 2^n)

hcell = cell(m, n);
for k = 1:m
    for j = 1:n
        if k == 1
            if j ~= 1
                hcell{k, j} = reshape(id, [1,d,d,1]);
            else
                hcell{k, j} = reshape(sz, [1,d,d,1]);
            end
        else
            if j == k || j == k - 1
                hcell{k, j} = reshape(sz, [1,d,d,1]);
            else
                hcell{k, j} = reshape(id, [1,d,d,1]);
            end
        end
    end
end

mpoI0 = hcell_to_mpo(hcell);
mpoI0 = mpo_compress(mpoI0, sv_min, Dc, 2);
mpoI0 = mpo_normalize(mpoI0);

I0_GS_energy = mps_overlap(mps0, mpo_mps(mpoI0, mps0)) * normI0




         
%% adiabatic evolution Hamiltonian
%I0 = Z1 + Z1Z1 + ... + Z4Z5, HP = X1 + ... + X5
%HT = f1 Z1 + ... + f5 Z5 + c12 Z1Z2 + ... + c45 Z4Z5
%Hadb = (1 - f) I0 + f HT + (1 - f)f HP

%single qubit terms
num_Hadb_1q_terms = 3; %I0: Z1, HT: Zj, HP: Xj

Hadb_1q = struct('sys', cell(num_Hadb_1q_terms, 1), 'op', ...
    cell(num_Hadb_1q_terms, 1));

for j = 1:num_Hadb_1q_terms
    Hadb_1q(j).sys = (1:n);
    Hadb_1q(j).op = cell(n, 1);
end

for j = 1:n
    if j == 1
        Hadb_1q(1).op{j} = sz;
    else
        Hadb_1q(1).op{j} = 0 * id;
    end
    Hadb_1q(2).op{j} = HT_frequencies(j) * sz;
    Hadb_1q(3).op{j} = sx;
end

%2 qubit terms
Hadb_2q = cell(n - 1, 2); %row1: I0, row2: HT
for j = 1:n-1
    Hadb_2q{j, 1} = kron(sz, sz);
    Hadb_2q{j, 2} = HT_couplings(string(j) + string(j + 1)) * kron(sz, sz);
end

%simulation parameters
dt = 0.1;
T_adb = 50 * pi;
time_steps_adb = round(T_adb / dt);
f_adb = linspace(0, 1, time_steps_adb);


%including the f_adb coefficient. (1-f) I0 + f Ht + f(1-f) HP
coeff_adb_1q = zeros(time_steps_adb, num_Hadb_1q_terms);
coeff_adb_1q(:, 1) = 1 - f_adb;
coeff_adb_1q(:, 2) = f_adb;
coeff_adb_1q(:, 3) = f_adb .* (1 - f_adb);
coeff_adb_1q = coeff_adb_1q(:); %flattens array

coeff_adb_2q = zeros(time_steps_adb, 2);
coeff_adb_2q(:, 1) = 1 - f_adb;
coeff_adb_2q(:, 2) = f_adb;
coeff_adb_2q = coeff_adb_2q(:);

%% adiabatic time evolution of MPO

[mpoIT, d_list] = mpo_evol(Hadb_2q, Hadb_1q, coeff_adb_2q, coeff_adb_1q, T_adb, mpoI0, tebd_options);

%need to decompose this over the set of Pauli operators to compare with 
%QuTIP simulation - only way to check!
%note that the energy changes as the T_adb increases because
%the weight of the non-Z strings decreases

IT_GS_energy = mps_overlap(mpsT, mpo_mps(mpoIT, mpsT)) * normI0




