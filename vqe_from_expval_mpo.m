
% Classical VQE by evolving the observable with tensor network methods
% for optimal quantum control

%imports
addpath("TiQC-ToQC/")
addpath("pauli_string_functions_module_matlab/")


%definitions and initial bond dimension / compression
q0 = [1; 0];
q1 = [0, 1];
id = eye(2);
sx = [0 1; 1 0];
sy = [0 -1i; 1i 0];
sz = [1 0; 0 -1];
tebd_options = tebd_default_options;
tebd_options.bond_dim = Inf;
tebd_options.bond_comp = 100;
sv_min = tebd_options.sv_min;
Dc = tebd_options.bond_comp;
tebd_options_mps = tebd_options;


%no. of qubits
n = 4;

%make sure H1q is correctly setup (in the following way for 2qubit control)
H1q = struct('sys', cell(2 * n, 1), 'op', cell(2 * n, 1));
for j = 1:n
    H1q(2 * j - 1).sys = j;
    H1q(2 * j).sys = j;
    
    H1q(2 * j - 1).op = {sx};
    H1q(2 * j).op = {sy};
end

%no 2 qubit control setup
H2q = cell(1, n - 1);
for j = 1:n - 1
    H2q{j} = -kron(sz, sz);
end


%intial state = |00...0> <00...0|
psi0 = computational_state_to_mpo(zeros(1, n));
psi0 = mpo_normalize(mpo_compress(psi0, sv_min, Dc, 2));


%observable - crushed Ising model 
% (\sumj Zj) + (\sum_k Xk Xk+1) + X1 + Xn
X1 = ['X' repmat('I', 1, n - 1)];
Xn = [repmat('I', 1, n - 1) 'X'];

observable_terms = [all_single_operator_strings(n, 'Z') X1 Xn local_2body_strings(n, 'XX')];
observable_weights = ones(length(observable_terms));


%observable - shortest vector style long-ranged ZZ
% (\sum_j Zj) + (\sum_j \sum_k~=j ZjZk)
observable_terms = [all_single_operator_strings(n, 'Z') nonlocal_2body_strings(n, 'Z', 'Z')];
observable_weights = [0.32547792479077287 1.4269790761690317 1.8885010855757431 ...
        0.8412581291193849 2.724066694375117 1.7834334919673296 0.5661473949021019 ...
        1.8471493437650164 1.8094761724022477 2.404681849201431];


%Hilbert-Schmidt norm
observable_norm = 0;
for j = 1:length(observable_terms)
    observable_norm = observable_norm + abs(observable_weights(j))^2;
end
observable_norm = sqrt(observable_norm * 2^n);

%summing MPOs
temp_mpos = pauli_strings_to_mpo(n, observable_terms, observable_weights);
observable_mpo = temp_mpos{1};
for j = 2:length(temp_mpos)
    observable_mpo = mpo_add(temp_mpos{j}, observable_mpo);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulation parameters
T = 2 * pi;
bin_num = 10;

dt = 0.01;
time_steps = round(T / dt);
times = linspace(0, T, time_steps);

ctrl_lb = -5; %for uniform constraints on the largest control pulses
ctrl_ub = 5;
varT = 0; %when varT set to 1 (off), no longer optimises?


%optimisation options (no constaints but need nloncon)
expval_target = -Inf;
options = optimoptions('fmincon', 'SpecifyObjectiveGradient', true, 'HessianApproximation','lbfgs','Display','iter');
options.StepTolerance = 1e-8;
options.ConstraintTolerance = 1e-8;
options.MaxFunctionEvaluations = 50;
options.ObjectiveLimit = expval_target;
A = [];
b = [];
Aeq = [];
beq = [];
nonlcon = [];

%function - choice of whether to use 2 qubit control or not
%fun = @(x) expval_2q_vqe(H2q, H1q, x, observable_mpo, psi0, varT, tebd_options);
fun = @(x) expval_vqe(H2q, H1q, x, observable_mpo, psi0, varT, tebd_options);
ctrl_num = length(H1q); %+ length(H2q);% if using 2 qubit control

%initial small test with random initial conditions 
Ntry = 4;
xL = cell(Ntry, 1);
expvalL = zeros(Ntry, 1);

parfor jtry = 1:Ntry
    c0 = rand() * (rand(bin_num, ctrl_num) - 1/2);
    x0 = [c0(:); T];

    lb = [ctrl_lb .* ones(1, length(x0) - 1) 0];
    ub = [ctrl_ub .* ones(1, length(x0) - 1) Inf];

    expvalL(jtry) = fun(x0);

    x_optm = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);

    xL{jtry} = x_optm;
    expvalL(jtry) = fun(x_optm);
end

[expval_min, jmin] = min(expvalL);
x_optm = xL{jmin};


%choose most promising initial state
options.MaxFunctionEvaluations = 500;
options.ObjectiveLimit = expval_target;

lb = [ctrl_lb .* ones(1, length(x_optm) - 1) 0];
ub = [ctrl_ub .* ones(1, length(x_optm) - 1) Inf];
x_optm = fmincon(fun, x_optm, A, b, Aeq, beq, lb, ub, nonlcon, options);
expval_optm = fun(x_optm);

fprintf('Expectation value: %d', expval_optm * observable_norm);

writematrix(x_optm, "vqe_n=4_.csv")