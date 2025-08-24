
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
tebd_options.bond_dim = 40;
tebd_options.bond_comp = 40;
sv_min = tebd_options.sv_min;
Dc = tebd_options.bond_comp;
tebd_options_mps = tebd_options;


%no. of qubits
n = 10;

%make sure H1q is correctly setup (in the following way )
H1q = struct('sys', cell(2 * n, 1), 'op', cell(2 * n, 1));
for j = 1:n
    H1q(2 * j - 1).sys = j;
    H1q(2 * j).sys = j;
    
    H1q(2 * j - 1).op = {sx};
    H1q(2 * j).op = {sy};
end

%no 2 qubit control setup (only native ZZ gates)
H2q = cell(1, n - 1);
for j = 1:n - 1
    H2q{j} = -1 * kron(sz, sz);
end


%intial state = |00...0> <00...0|
psi0 = computational_state_to_mpo(zeros(1, n));
psi0 = mpo_normalize(mpo_compress(psi0, sv_min, Dc, 2));


%observable - crushed Ising model 
% (\sumj Zj) + (\sum_k Xk Xk+1) + X1 + Xn
% X1 = ['X' repmat('I', 1, n - 1)];
% Xn = [repmat('I', 1, n - 1) 'X'];
% observable_terms = [all_single_operator_strings(n, 'Z') X1 Xn local_2body_strings(n, 'XX')];
% observable_weights = ones(length(observable_terms));

%observable - shortest vector style long-ranged ZZ
% (\sum_j Zj) + (\sum_j \sum_k~=j ZjZk)
 observable_terms = [all_single_operator_strings(n, 'Z') nonlocal_2body_strings(n, 'Z', 'Z')];
 observable_weights = [1.1936187224736132,
 3.942825143869964,
 3.6444516935156717,
 4.7924115711654105,
 1.6379460719285204,
 2.987007872098033,
 4.625704903118575,
 3.6385001168980953,
 1.573522320013755,
 0.139197506979587,
 2.8203807481050003,
 4.7385461744266495,
 3.9414186597085665,
 1.1470288101844017,
 4.645067306076715,
 3.650233413855497,
 4.367913839287504,
 4.9118091882909924,
 1.2590291636070035,
 2.2845950442155853,
 3.668678878572287,
 1.5456911234957889,
 1.6979930999985422,
 3.3754463529892496,
 4.556308027300372,
 1.1651963864263122,
 4.112484070760469,
 4.650565980053052,
 2.0214625070401047,
 3.6740895399130276,
 1.0343769942154202,
 2.27824051442235,
 0.8147232044255909,
 2.082441854259157,
 4.000397248674186,
 4.054545366338213,
 3.123007478477453,
 0.8871699028963204,
 4.593817444629671,
 2.8444035537849084,
 4.63911019539531,
 0.22095250449073123,
 3.5321701099412577,
 2.126535044880434,
 3.515545287340134,
 3.713074499360987,
 1.871845269181085,
 4.504410332691487,
 3.2915313987321064,
 3.469092312581874,
 2.539319103121453,
 1.9965462822785613,
 3.1546998198967096,
 1.0399826111302208,
 0.8996954811401109];

%observable - Heisenberg model at criticality (highly entangled GS)
% \sum_j XjXj+1 + YjYj+1 + ZjZj+1
%observable_terms = [local_2body_strings(n, 'XX') local_2body_strings(n, 'YY') local_2body_strings(n, 'ZZ')];
%observable_weights = ones(length(observable_terms));



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
T = 4 * pi;
bin_num = 10;
    
dt = 0.001;
time_steps = round(T / dt);
times = linspace(0, T, time_steps);

ctrl_lb = -5; %for uniform constraints on the largest control pulses
ctrl_ub = 5;



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

varT = 0; %when varT set to 1 (off), no longer optimises?
fun = @(x) expval_vqe(H2q, H1q, x, observable_mpo, psi0, varT, tebd_options);
ctrl_num = length(H1q);% + length(H2q);% if using 2 qubit control

%initial small test with random initial conditions 
Ntry = 4;
xL = cell(Ntry, 1);
expvalL = zeros(Ntry, 1);
intial_guess = cell(Ntry, 1);
parfor jtry = 1:Ntry
    c0 = rand() * (rand(bin_num, ctrl_num) - 1/2);
    initial_guess{jtry} = c0;
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

%writematrix(x_optm, "heis_n=4_bin=1.csv")
%%
% why is this not fking working!!!
%x_optm_py = [ 0.17686671,  0.08839826,  0.06544213,  0.23551391, -0.09255747,...
%       -0.07200062, -0.18091324, -0.23299775,  0.21637721,  0.24563825,...
%        0.15802555,  0.072183  , -0.27337751, -0.24380614, -0.21329594,...
%        0.01071987, 6.36462444991919];

%[obs_exp_val, gradienty] = expval_vqe(H2q, H1q, x_optm_py, observable_mpo, psi0, varT, tebd_options);
%different!!! obs_exp_val different to the final expecation value
%(-5.24999)


