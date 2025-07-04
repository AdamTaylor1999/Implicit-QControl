
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
addpath("pauli_string_functions_module_matlab/")


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
sv_min = tebd_options.sv_min;
Dc = tebd_options.bond_comp;

%no. of qubits
n = 4;

%initial invariants: initially just considering single Pauli strings, not
%sums or combinations
I0_strings = [all_single_operator_strings(n, 'YY'), local_2body_strings(n, 'ZZ')];
m = length(I0_strings);
I0_mpos = pauli_strings_to_mpo(n, I0_strings);
for j = 1:m
    I0_mpos{j} = mpo_normalize(mpo_compress(I0_mpos{j}, sv_min, Dc, 2));
end


%Heisenberg Hamiltonian: H_heis = XjXj+1 + YjYj+1 + ZjZj+1
%POTENTIAL REACHABILITY ISSUE WITH {X, ZZ} CONTROLS

H2q_heis = cell(n - 1, 1);
for j = 1:n - 1
    H2q_heis{j, 1} = kron(sy, sy) + kron(sz, sz);
end

%time evolution parameters
dt = 0.01;
T_evol = 1;
time_steps = round(T_evol / dt);
times = linspace(0, T_evol, time_steps);

%two qubit Hamiltonian's time dependence (constant)
c2q = ones(1, time_steps);

%single qubit Hamiltonian - set to zero to not break time evolution pacakge
H1q = struct('sys', cell(1, 1), 'op', cell(1,1));
H1q.sys = (1:n);
H1q.op = cell(n, 1);
for j = 1:n
    H1q.op{j} = id;
end
c1q = zeros(time_steps, 1);

%time evolution for Ij0 --> IjT to find target MPOs
IT_mpos = cell(1, m);
for j = 1:m
    [IT_mpos{j}, ~] = mpo_evol(H2q_heis, H1q, c2q, c1q, T_evol, I0_mpos{j}, tebd_options);
end

%decomposing target MPOs over Pauli strings (used for verification)
pauli_strings = all_pauli_strings(n);
IT_mpos_decomp = cell(1, m);
for j = 1:m
    IT_mpos_decomp{j} = mpo_pauli_decomposition(n, pauli_strings, IT_mpos{j});
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%optimisation - remember, we just need the relative values of everything!
tebd_options.bond_dim = Inf;
tebd_options_mps = tebd_options;
tebd_options_mps.bond_comp = 100;

%define control Hamiltonian; H(t) = \sum_j fj Xj + cj ZjZj+1

%MAKING UNIVERSAL CONTROL {X, Z, ZZ} --> still 10% infidelity?
%also SOMETIMES (33%) gives an error on mpo_overlap (Specified dims of the
%input arrays must have the same size) from infid_2q gradient calculation

%2 qubit terms
H2q = cell(1, n - 1); %different structure for some reason? based on example
for j = 1:n - 1
    H2q{j} = kron(sz, sz);
end
%1 qubit terms, controllable. Now include X and Z (universal?)
H1q = struct('sys', cell(n, 1), 'op', cell(n, 1));
for j = 1:n
    H1q(j).sys = j;
    H1q(j).op = {sx};
    %H1q(n + j).sys = j; %BEWARE - THIS SEEMINGLY INTRODUCES ERRORS
    %H1q(n + j).op = {sy};
end
ctrl_num = length(H1q) + (n - 1);




%sets course-ness of control operations and timescale. Scales as a function
%of qubit number (WHY?)
bin_factor = 10;
bin_num = n * bin_factor;
duration_factor = 5 * pi;
T0 = n * duration_factor;
varT = 0; %allows for optimisation of control evolution length.
% Set to zero while we test optimising with two qubit controls (infid_2q)



%this part will need fixing in the mean time, we just look at a single
%invariant to test the basic optimisation

fun = @(x) infid_2q_propagator(H2q, H1q, x, I0_mpos, IT_mpos, varT, tebd_options);

%optimisation options
iF_target = 1e-3;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];

%initially try 4 different random initial states for limited run
options = optimoptions('fmincon', 'SpecifyObjectiveGradient', true,'HessianApproximation','lbfgs','Display','iter');
options.StepTolerance = 1e-6;
options.ConstraintTolerance = 1e-8;
options.MaxFunctionEvaluations = 50;
options.ObjectiveLimit = iF_target;

nonlcon = [];
Ntry = 4;
xL = cell(Ntry, 1);
iFL = zeros(Ntry, 1);

parfor jtry = 1:Ntry
    c0 = rand() * (rand(bin_num, ctrl_num) - 1/2)
    x0 = [c0(:); T0]; %T0 NOT BEING OPTIMISED, SET TO T0 FOR NOW
    disp(length(x0))
    iF0 = fun(x0)
    x_optm = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    xL{jtry} = x_optm;
    iFL(jtry) = fun(x_optm);
end

[iFmin, jmin] = min(iFL);
x_optm = xL{jmin};

%choose most promising initial state and use this to optimise
options.MaxFunctionEvaluations = 500;
options.ObjectiveLimit = iF_target;

x_optm = fmincon(fun, x_optm, A, b, Aeq, beq, lb, ub, nonlcon, options);
iF_optm = fun(x_optm);
fprintf('Operator infidelity %d\n', iF_optm);

%
%post-optimisation - extracting the optimal control coefficients 
T_optm = x_optm(end);
c = x_optm(1:end - 1);
c = reshape(c, [bin_num, ctrl_num]);
c1q_binned = c(:, 1:n);
c2q_binned = c(:, n + 1:2 * n - 1);

dt = 0.1;
time_steps = round(T_optm / dt);
times = linspace(0, T_optm, time_steps);

c1q = cell(time_steps, n);
bin_size = T_optm / bin_num;
count = 1;
for j = 1:time_steps
    if times(:, j) > count * bin_size
        count = count + 1;
    end
    for k = 1:n
        c1q{j, k} = c1q_binned(count, k);
    end
end
c1q = c1q(:);
    
 
H1q = struct('sys', cell(n, 1), 'op', cell(n, 1));
for j = 1:n
    H1q(j).sys = (1:n);
    H1q(j).op = cell(n, 1);
    for k = 1:n
        if k == j
            H1q(j).op{k} = sx;
        else
            H1q(j).op{k} = 0 * id;
        end
    end
end

%how do we add independent time-dependence of the two-qubit gates?
H2q = cell(n - 1,  1);
for j = 1:n - 1
    H2q{j, 1} = kron(sz, sz);
end



%IT_mpos_optm = cell(1, m);
%for j = 1:m
%    [IT_mpos_optm, ~] = mpo_evol(H2q, H1q, c2q_ctrl, c1q_ctrl, T_ctrl, I0_mpos{j}, tebd_options);
%end




% issue; fmincon ending optimisation early
% "fmincon stopped because the size of the current step is less than the
% value of the step size tolerance and constraints are satisfied to within
% the value of the constraint tolerance.

%Perhaps we are terminating at the minimum (iF = 3.745519e-1) but it is not
%reachable? 
%Seems to imply that we can get two of the three Pauli terms working as
%fidelity 66%. Which terms are we getting? Let's find out what the target
%is, and what it should actually be .
% I wonder if the 2body interactions are working properly!
% Maybe we are actually optimising just using single qubit X gates (which
% would explain why the result is approximately 1/3, since we can only
% really get the X terms exactly right, not the Y or Z terms.

