%compare
addpath("TiQC-ToQC/")
addpath("pauli_string_functions_module_matlab/")

nw=4;
maxNumCompThreads(nw);
%parpool('local',nw)
d=2;
q0=[1;0];
q1=[0;1];
qm=(q0-q1)/sqrt(2);
qp=(q0+q1)/sqrt(2);
sx=[0,1;1,0]; 
sy=[0,-1i;1i,0]; 
sz=[1,0;0,-1]; 
id=eye(2);
s_plus=(sx+1i*sy)/2;
s_minus=(sx-1i*sy)/2;
no=(id-sz)/2;%number operator 
n=5
%tebd options
tebd_options=tebd_default_options;
tebd_options.bond_dim=30;
tebd_options.bond_comp=10;
sv_min=tebd_options.sv_min;
Dc=tebd_options.bond_comp;
%mps for verification 
mps0=cell(1,n);
for j=1:n
mps0{j}=reshape(qm,[1,d,1]);
end
mps0=mps_normalize(mps0);
mpstg_exact=cell(1,n);
for j=1:n
mpstg_exact{j}=reshape(q0,[1,d,1]);
end
for j=1:2:n
mpstg_exact{j}=reshape(q1,[1,d,1]);
end
mpstg_exact=mps_normalize(mpstg_exact);
%initial mpo %this is sum x_j
m=n;
hcell=cell(m,n); 
for k=1:m
for j=1:n
hcell{k,j}=reshape(id,[1,2,2,1]);
end
hcell{k,k}=reshape(sx,[1,2,2,1]);
end
mpo0=hcell_to_mpo(hcell);
norm0=n*2^n;
mpo0=mpo_compress(mpo0,sv_min,Dc,2);
mpo0=mpo_normalize(mpo0);
%check energy
mps1=mpo_mps(mpo0,mps0);
erg0_mpo=mps_overlap(mps1,mps0)*sqrt(norm0);

%Adiabatic evolution
fprintf('adiabatic evolution\n');
%define hamiltonian terms H(t)=c2q(t)H2q+c(t)Hc
H2q=cell(n-1,1);
for j =1:n-1
    H2q{j,1}=kron(no,no);   %H2q=\sum_j n_j n_{j+1}
end
ctrl_num_adb=2;
Hc=struct('sys',cell(ctrl_num_adb,1),'op',cell(ctrl_num_adb,1));
count=1;
Hc(count).sys=(1:n);
Hc(count).op=cell(n,1);
for j=1:n
    Hc(count).op{j}=no;   %Hc1=\sum_j n_j
end
count=2;
Hc(count).sys=(1:n);      %Hc2=\sum_j x_j
Hc(count).op=cell(n,1);
for j=1:n
    Hc(count).op{j}=sx;
end                       %H(t)=c2q(t) H2q + c1(t) Hc1 + c2(t) Hc2
%pulse
dt=0.1;%time step
Tadb=100*pi;%duration
bin_num_adb=round(Tadb/dt);%number of time bins
f=linspace(0,1,bin_num_adb)';
c2q=f;
c=zeros(bin_num_adb,ctrl_num_adb);
c(:,1)=-f;
c(:,2)=2*(1-f);
c=c(:);%vectorization
[mpo,~]=mpo_evol(H2q,Hc,c2q,c,Tadb,mpo0,tebd_options);
mps1=mpo_mps(mpo,mpstg_exact);
erg=real(mps_overlap(mpstg_exact,mps1))*sqrt(norm0);
Fbound=1-abs(real(erg)+n)/2;
fprintf('system size #%d\n',n);
fprintf('energy per site of control target %d\n',erg/n);
fprintf('fidelity bound for ground state of control target %d\n',Fbound);
mpo_adb_target=mpo;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%optimization
fprintf('optimization\n');
tebd_options.bond_dim=Inf;
tebd_options_mps=tebd_options;
tebd_options_mps.bond_comp=100;
%Define control Hamiltonian
H2q=cell(1,n-1);%two-qubit constant terms
for j =1:n-1
    H2q{j}=kron(no,no);
end
ctrl_num=2*n;
%single-qubit terms

Hc=struct('sys',cell(ctrl_num,1),'op',cell(ctrl_num,1));
count=0;
for kk=1:n
count=count+1;
Hc(count).sys=[kk];   %H(t)=H2q+\sum_j f_j(t) Z_j + \sum_j g_j X_j
Hc(count).op={sz};  
count=count+1;
Hc(count).sys=[kk];
Hc(count).op={sx};
end



bin_factor=10;
duration_factor=pi;
T0=n*duration_factor; 
bin_num=n*bin_factor;
varT=1; %whether to optimize the duration
fun=@(x) infid(H2q,Hc,x,mpo0,mpo_adb_target,varT,tebd_options);
%optimization options
iF_target=10^(-3);
A=[];
b=[];
Aeq=[];
beq=[];
lb=[];
ub=[];
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'HessianApproximation','lbfgs','Display','iter');
options.MaxFunctionEvaluations = 50;
options.ObjectiveLimit=iF_target;
nonlcon=[];
Ntry=4;
xL=cell(Ntry,1);
iFL=zeros(Ntry,1);
parfor jtry=1:Ntry
c0=rand()*(rand(bin_num,ctrl_num)-1/2);
x0=[c0(:);T0*rand()];
iF0=fun(x0)
x_optm=fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
xL{jtry}=x_optm;
iFL(jtry)=fun(x_optm);
end
[iFmin,jmin]=min(iFL);
x_optm=xL{jmin};
options.MaxFunctionEvaluations = 500;
options.ObjectiveLimit=iF_target;
%refinement
x_optm=fmincon(fun,x_optm,A,b,Aeq,beq,lb,ub,nonlcon,options);
iF_optm=fun(x_optm);
fprintf('operator infidelity %d\n',iF_optm);
%state infidelity check
iF_state=mps_infid_nograd(H2q,Hc,x_optm,mps0,mpstg_exact,tebd_options_mps);
fprintf('state infidelity %d\n',iF_state);
save mis_nb_optm_dat
