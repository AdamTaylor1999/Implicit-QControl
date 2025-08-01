function [iF,iG]=infid_2q(H2q,H1q,x,mpo0,mpotg,varT,tebd_options)

H0 = H2q;
Hc = H1q;
%Adjusting the function infid to allow for control of two qubit gates.

%H0 --> H2q (n-1 terms). H2q = {s1s2, s2s3, s3s4, ..., sn-1sn}
%Hc --> H1q (n terms)
% x = c1, c2, ... cn, T --> x = c1, c1, .. ,cn, cn+1, ..., c2n-1, T
%x = control coefficients - assumed we can vary the number for H1q, but
%have individual control for all n-1 2 qubit gates


%calculates the infidelity and gradient 

sv_min=tebd_options.sv_min;
D=tebd_options.bond_dim;
Dc=tebd_options.bond_comp;
nsweep=tebd_options.num_sweep;
midstep=tebd_options.num_midstep;
nt=tebd_options.num_refined_step;
iscpr=tebd_options.is_compressed;
iso=tebd_options.is_second_order;

n=length(mpo0);
nc=length(Hc);
nc2 = n - 1; %how many 2qubit gates we are controling (careful here!)

c = x(1:end - 1);%single qubit control copefficients (bin H dependence)
nbin = length(c) / (nc + nc2);


T=x(end);%total time
c = reshape(c, [nbin, nc + nc2]);

c1 = c(:, 1:nc);%single qubit control terms
c2 = c(:, nc + 1:end);%two qubit control terms (limited to controling only one two-qubit interaction per neighbour)


c = c1; %kept c1 as c for ease of implementation - will improve later
%nbin=length(c)/(nc); %no but we need to calculate nbin properly


d=size(mpo0{1},2);%local dimension
dt=T/(nbin*nt); %fine grain time step of TEBD simulation
Dt=T/nbin;%course grained time step at which the control Hamiltonians change

%half time evolution is for second order TEBD (iso = is_second_order).
%TEBD_default_settings set iso = 0 so not important for now. Will want to
%include this properly later when extending the code

if iso == 1
    error('Function not yet set up for second order trotter!')
end
%if iso==1
%    %initial half time evolution
%g20=cell(1,n-1);
%for j=1:n-1
%        h=H0{j};
%        gate=expm(1i*dt*h/2);%half time, backward
%        gate=reshape(gate,[d,d,d,d]);
%        g20{j}=gate;
%end
%%apply odd terms
%for j=1:2:n-1
%    [mpo0{j},mpo0{j+1}]=gate_2q_LR(mpo0{j},mpo0{j+1},g20{j},sv_min,D);
%end
% %apply even terms
%for j=2:2:n-1
%    [mpo0{j},mpo0{j+1}]=gate_2q_LR(mpo0{j},mpo0{j+1},g20{j},sv_min,D);
%end
%%backward
%%apply even terms
%for j=2:2:n-1
% %   [mpotg{j},mpotg{j+1}]=gate_2q_LR(mpotg{j},mpotg{j+1},g20{j},sv_min,D);
%end
%%apply odd terms
%for j=1:2:n-1
%    [mpotg{j},mpotg{j+1}]=gate_2q_LR(mpotg{j},mpotg{j+1},g20{j},sv_min,D);
%end
%if iscpr==1
%      mpo0=mpo_compress(mpo0,sv_min,Dc,nsweep);
%      mpotg=mpo_compress(mpotg,sv_min,Dc,nsweep);
%end
%mpo0=mpo_normalize(mpo0);
%mpotg=mpo_normalize(mpotg);
%end

mpofw=cell(nbin+1,n);  
mpobw=cell(nbin+1,n);
iG=zeros(nbin,nc + nc2);
%%%%%%%%%%%%%%%%%%%%%%%%

mpofw(1,:)=mpo0;
mpobw(1,:)=mpotg;

%forward 2qubit time unitary, eg g2 = {e^{-i dt x1x2}, e^{-i dt x2x3}, ...}
%g2=cell(1,n-1);
%for j=1:n-1
%        h=H0{j};
%        gate=expm(-1i*dt*h);
%        gate=reshape(gate,[d,d,d,d]);
%        g2{j}=gate;
%end
%backwards 2qubit time unitary (g2 but with dt --> -dt)
%g2bw=cell(1,n-1);
%for j=1:n-1
%        h=H0{j};
%        gate=expm(1i*dt*h);
%        gate=reshape(gate,[d,d,d,d]);
%        g2bw{j}=gate;
%end

%time dependent gate construction
for k=1:nbin
    %gate construction
    %1qubit time unitary (depends explicitly on time bin, k

    %2qubit forwards
    g2 = cell(1, n - 1);
    for j = 1:n - 1
        h = c2(k, j) * H0{j};
        gate = expm(-1i * dt * h);
        gate = reshape(gate, [d,d,d,d]);
        g2{j} = gate;
    end
    
    %1qubit control
    g1=cell(1, n);
    for j=1:n %which qubit, j
       h=zeros(d); %?
       for jc=1:nc %which control string are we considering 
            for js=1:length(Hc(jc).sys) %for js = 1:no. of qubits in this control string
                if Hc(jc).sys(js)==j
                    h=h+c(k,jc)*Hc(jc).op{js}; %
                end
            end
       end
       gate=expm(-1i*dt*h);
       g1{j}=gate;
    end 
    %forward propagation
   mpofw(k+1,:)=mpofw(k,:);
  
    for jt=1:nt
    
    for j=1:2:n-1
        [mpofw{k+1,j},mpofw{k+1,j+1}]=gate_2q_LR(mpofw{k+1,j},mpofw{k+1,j+1},g2{j},sv_min,D);
    end
     %apply even 2q terms
    for j=2:2:n-1
        [mpofw{k+1,j},mpofw{k+1,j+1}]=gate_2q_LR(mpofw{k+1,j},mpofw{k+1,j+1},g2{j},sv_min,D);
    end
    %apply 1q terms
    for j=1:n
        [mpofw{k+1,j}]=gate_1q_LR(mpofw{k+1,j},g1{j});
    end
    end


    %UP TO HERE!!!!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%backward propagation
%construct gates
%2qubit backwards 
    g2bw = cell(1, n - 1);
    for j = 1:n - 1
        h = c2(nbin - k  + 1, j) * H0{j};
        gate = expm(1i * dt * h);
        gate = reshape(gate, [d,d,d,d]);
        g2bw{j} = gate;
    end
%1qubit backwards
g1=cell(1,n);
    for j=1:n
        h=zeros(d);
        for jc=1:nc
            for js=1:length(Hc(jc).sys)
                if Hc(jc).sys(js)==j
        h=h+c(nbin-k+1,jc)*Hc(jc).op{js};
                end
            end
        end
        gate=expm(1i*dt*h);
        g1{j}=gate;
    end
   mpobw(k+1,:)=mpobw(k,:);
    %apply 1q terms
    for j=1:n
        [mpobw{k+1,j}]=gate_1q_LR(mpobw{k+1,j},g1{j});
    end

    %%%%%%%%%%%%%%%%%%%%%% UP TO HERE -- > WHY DO WE DO JT = 1:NT, DOESN'T
    %%%%%%%%%%%%%%%%%%%%%% APPEAR ANYWHERE!!!!!
    for jt=1:nt
    %apply even 2q terms
    for j=2:2:n-1
        [mpobw{k+1,j},mpobw{k+1,j+1}]=gate_2q_LR(mpobw{k+1,j},mpobw{k+1,j+1},g2bw{j},sv_min,D);
    end
    %apply odd  2q terms
    for j=1:2:n-1
        [mpobw{k+1,j},mpobw{k+1,j+1}]=gate_2q_LR(mpobw{k+1,j},mpobw{k+1,j+1},g2bw{j},sv_min,D);
    end
    end
    %compression (if appliable) and normalisation
if mod(k-1,midstep)==0||k==nbin
    if iscpr==1
     mpofw(k+1,:)=mpo_compress(mpofw(k+1,:),sv_min,Dc,nsweep);
     mpobw(k+1,:)=mpo_compress(mpobw(k+1,:),sv_min,Dc,nsweep);
    end
mpofw(k+1,:)=mpo_normalize(mpofw(k+1,:));
mpobw(k+1,:)=mpo_normalize(mpobw(k+1,:));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gradient
for k=1:nbin
    tnsfw_temp=mpofw(k+1,:);%unperturbed forward mpo. We then perturb it along each parameter direction
    tnsbw_temp=mpobw(nbin-k+1,:);
    for jc=1:nc  
           ovl_diff_left=0;
           for js=1:length(Hc(jc).sys)
               jq = Hc(jc).sys(js);
               gate = -1i*Dt*Hc(jc).op{js};
               tnsfw_diff_left = tnsfw_temp;
               tnsfw_diff_left{jq} = gate_1q(tnsfw_diff_left{jq},gate);
               ovl_diff_left = ovl_diff_left+mpo_overlap(tnsbw_temp,tnsfw_diff_left);
           end
           iG(k, jc)=-2*real(ovl_diff_left);
    end

    for jc2 = 1:nc2 %left as contracting from the left (hence tnsfw{j+1} = [] after contraction)
        gate = -1i * Dt * H0{jc2};
        gate = reshape(gate, [d,d,d,d]);

        tnsfw_diff_left = tnsfw_temp;
        


        [tnsfw_diff_left{jc2}, tnsfw_diff_left{jc2 + 1}] = gate_2q(tnsfw_diff_left{jc2}, tnsfw_diff_left{jc2+1}, gate, sv_min, D);
 
        %tnsfw_diff_left{jc2+1} = []; %mark j+1 as merged (mr chatGPT suggestion)
        
       
        
        ovl_diff_left = mpo_overlap(tnsbw_temp, tnsfw_diff_left);
        
        iG(k, nc + jc2) = -2 * real(ovl_diff_left);
    end 

end
ovl=mpo_overlap(mpotg,mpofw(nbin+1,:));
iF=1-real(ovl);
iG=iG(:);

%currently, set varT = 0 to avoid having to work with infid_nograd func
% as not yet updated to allow time dependent two qubit interactions.
%if varT==1
%dt=10^(-10);    
%x(end)=T+dt;
%iF1=infid_nograd(H0,Hc,x,mpo0,mpotg,tebd_options);
%iGT=(iF1-iF)/dt;
%else
%iGT=0;
%end
iGT = 0;
iG=[iG;iGT];
end
