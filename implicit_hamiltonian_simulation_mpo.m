
% Quantum control for realising propagators
%
% In this code, we use quantum invariants to find quantum control
% solutions for Hamiltonian simulation. 



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
I0_strings = [all_single_operator_strings(n, 'Y'), local_2body_strings(n, 'ZZ')];
m = length(I0_strings);
I0_mpos = pauli_strings_to_mpo(n, I0_strings);
for j = 1:m
    I0_mpos{j} = mpo_normalize(mpo_compress(I0_mpos{j}, sv_min, Dc, 2));
end


%Heisenberg Hamiltonian: H_heis = XjXj+1 + YjYj+1 + ZjZj+1
%REACHABILITY ISSUE WITH {X, ZZ} CONTROLS

%want control operators that have Lie algebra featuing XX, YY, ZZ BUT IS
%NOT universal

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

%2 qubit terms
H2q = cell(1, n - 1); %different structure for some reason? based on example
for j = 1:n - 1
    H2q{j} = kron(sz, sz);
end
%1 qubit terms, controllable. Now include X and Y (universal)
H1q = struct('sys', cell(n, 1), 'op', cell(n, 1));%SEEMS TO SPEND AGES OPTIMISING VERY VERY SLOWLY. MAYBE AN ISSUE WITH THE CODE? MAYBE AN ISSUE INHERENT ELSEWHERE (eg; universal so optimisation slow???)?
for j = 1:n
    H1q(j).sys = j;
    %H1q(n + j).sys = j;
    H1q(j).op = {sx};
    %H1q(n + j).op = {sy};
    %H1q(n + j).sys = j; %BEWARE - THIS SEEMINGLY INTRODUCES ERRORS. WORK
    %OUT WHY!!!!!!
    %H1q(n + j).op = {sy}; -- seemingly gone?
end
ctrl_num = length(H1q) + (n - 1);




%sets course-ness of control operations and timescale. Scales as a function
%of qubit number (WHY?)
bin_factor = 10;
bin_num = n * bin_factor;
duration_factor = 4 * pi;
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
    iF0 = fun(x0)
    x_optm = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    xL{jtry} = x_optm;
    iFL(jtry) = fun(x_optm);
end

[iFmin, jmin] = min(iFL);
x_optm = xL{jmin};

%choose most promising initial state and use this to optimise
options.MaxFunctionEvaluations = 1000;
options.ObjectiveLimit = iF_target;

x_optm = fmincon(fun, x_optm, A, b, Aeq, beq, lb, ub, nonlcon, options);
iF_optm = fun(x_optm);
fprintf('Operator infidelity %d\n', iF_optm);

%writematrix(x_optm, "... .csv")
