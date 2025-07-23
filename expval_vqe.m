function [iF,iG]=expval_vqe(H0,Hc,x,mpo0,mpotg,varT,tebd_options)

%What form must Hc take? Can it include two-qubit gates?
%lternatively, can we define multiple H0's with multiple c0's? This is what
%needs to be done, which may be a significant rewrite of the function

%calculates the infidelity and gradient 

T=x(end);%total time

c=x(1:end-1);%control coefficients

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
nbin=length(c)/(nc);
c = reshape(c, [nbin, nc]);%reshaped into individual 
d=size(mpo0{1},2);
dt=T/(nbin*nt);
Dt=T/nbin;
if iso==1
    %initial half time evolution
g20=cell(1,n-1);
for j=1:n-1
        h=H0{j};
        gate=expm(1i*dt*h/2);%half time, backward
        gate=reshape(gate,[d,d,d,d]);
        g20{j}=gate;
end
%apply odd terms
for j=1:2:n-1
    [mpo0{j},mpo0{j+1}]=gate_2q_LR(mpo0{j},mpo0{j+1},g20{j},sv_min,D);
end
 %apply even terms
for j=2:2:n-1
    [mpo0{j},mpo0{j+1}]=gate_2q_LR(mpo0{j},mpo0{j+1},g20{j},sv_min,D);
end
%backward
%apply even terms
for j=2:2:n-1
    [mpotg{j},mpotg{j+1}]=gate_2q_LR(mpotg{j},mpotg{j+1},g20{j},sv_min,D);
end
%apply odd terms
for j=1:2:n-1
    [mpotg{j},mpotg{j+1}]=gate_2q_LR(mpotg{j},mpotg{j+1},g20{j},sv_min,D);
end
if iscpr==1
      mpo0=mpo_compress(mpo0,sv_min,Dc,nsweep);
      mpotg=mpo_compress(mpotg,sv_min,Dc,nsweep);
end
mpo0=mpo_normalize(mpo0);
mpotg=mpo_normalize(mpotg);
end

mpofw=cell(nbin+1,n);  
mpobw=cell(nbin+1,n);
iG=zeros(nbin,nc);
%%%%%%%%%%%%%%%%%%%%%%%%
mpofw(1,:)=mpo0;
mpobw(1,:)=mpotg;

g2=cell(1,n-1);
for j=1:n-1
        h=H0{j};
        gate=expm(-1i*dt*h);
        gate=reshape(gate,[d,d,d,d]);
        g2{j}=gate;
end
g2bw=cell(1,n-1);
for j=1:n-1
        h=H0{j};
        gate=expm(1i*dt*h);
        gate=reshape(gate,[d,d,d,d]);
        g2bw{j}=gate;
end

for k=1:nbin
    %single qubit gate construction
    g1=cell(1,n);
    for j=1:n
       h=zeros(d);
       for jc=1:nc
            for js=1:length(Hc(jc).sys)
                if Hc(jc).sys(js)==j
        h=h+c(k,jc)*Hc(jc).op{js}; %leads to h = sum_j cj(t) H_j for single qubit H_j
                end
            end
       end
       gate=expm(-1i*dt*h);
       g1{j}=gate;
    end 
    %forward propagation
   mpofw(k+1,:)=mpofw(k,:);
    for jt=1:nt 
     %apply odd 2q terms
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%backward propagation
%construct gates
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
    %normalisation
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
    tnsfw_temp=mpofw(k+1,:);
    tnsbw_temp=mpobw(nbin-k+1,:);
    for jc=1:nc  
            ovl_diff_left=0;
           for js=1:length(Hc(jc).sys)
                 jq=Hc(jc).sys(js);
                 gate=-1i*Dt*Hc(jc).op{js};
                 tnsfw_diff_left=tnsfw_temp;
            tnsfw_diff_left{jq}=gate_1q(tnsfw_diff_left{jq},gate);
            ovl_diff_left=ovl_diff_left+mpo_overlap(tnsbw_temp,tnsfw_diff_left);
           end
           %CHANGED GRADIENT SIGN
              iG(k,jc)=2*real(ovl_diff_left);
    end
end
ovl=mpo_overlap(mpotg,mpofw(nbin+1,:));
%CHANGED iF = 1 - ovl --> ovl
iF=real(ovl);
iG=iG(:);
if varT==1
dt=10^(-10);    
x(end)=T+dt;
iF1=infid_nograd(H0,Hc,x,mpo0,mpotg,tebd_options);
iGT=(iF1-iF)/dt;
else
iGT=0;
end
iG=[iG;iGT];
end
