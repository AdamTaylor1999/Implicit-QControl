function [iFlist,Dlist]=mps_infid_T_nograd_robust_prl(J0,H00,Hu0,Hc0,x,mps00,mpstg0,ismean,tebd_options)
sv_min=tebd_options.sv_min;
D=tebd_options.bond_dim;
Dc=tebd_options.bond_comp;
nsweep=tebd_options.num_sweep;
midstep=tebd_options.num_midstep;
nt=tebd_options.num_refined_step;
iscpr=tebd_options.is_compressed;
iso=tebd_options.is_second_order;
c0=x(1:end-1);
T=x(end);
n=length(mps00);
nc=length(Hc0);
nbin=length(c0)/(nc);
c0=reshape(c0,[nbin,nc]);
d=size(mps00{1},2);
dt=T/(nbin*nt);
nsampl=size(J0,1);
iFlist=zeros(1,nsampl);
Dlist=zeros(nsampl,1);
parfor jsampl=1:nsampl
H0=H00;
Hc=Hc0;
Hu=Hu0;
num_int=size(Hu,2);
J=J0;
mps0=mps00;
mpstg=mpstg0;
c=c0;
maxD=1;
if iso==1
    %initial half time evolution
g20=cell(1,n-1);
for j=1:n-1
       h=0;
    for l=1:num_int
        h=h+J(jsampl,j,l)*Hu{j,l};
    end
    h=h+H0{j};
        gate=expm(1i*dt*h/2);%half time, backward
        gate=reshape(gate,[d,d,d,d]);
        g20{j}=gate;
end
%apply odd terms
for j=1:2:n-1
    [mps0{j},mps0{j+1}]=mps_gate_2q(mps0{j},mps0{j+1},g20{j},sv_min,D);
end
 %apply even terms
for j=2:2:n-1
    [mps0{j},mps0{j+1}]=mps_gate_2q(mps0{j},mps0{j+1},g20{j},sv_min,D);
end
%backward
%apply even terms
for j=2:2:n-1
    [mpstg{j},mpstg{j+1}]=mps_gate_2q(mpstg{j},mpstg{j+1},g20{j},sv_min,D);
end
%apply odd terms
for j=1:2:n-1
    [mpstg{j},mpstg{j+1}]=mps_gate_2q(mpstg{j},mpstg{j+1},g20{j},sv_min,D);
end
if iscpr==1
      mps0=mps_compress(mps0,sv_min,Dc,nsweep);
      mpstg=mps_compress(mpstg,sv_min,Dc,nsweep);
end
mps0=mps_normalize(mps0);
mpstg=mps_normalize(mpstg);
end
mps=mps0;
g2=cell(1,n-1);
for j=1:n-1
        h=0;
    for l=1:num_int
        h=h+J(jsampl,j,l)*Hu{j,l};
    end
    h=h+H0{j};
        gate=expm(-1i*dt*h);
        gate=reshape(gate,[d,d,d,d]);
        g2{j}=gate;
end
for k=1:nbin
     %gate construction
   g1=cell(1,n);
    for j=1:n
       h=zeros(d);
       for jc=1:nc
            for js=1:length(Hc(jc).sys)
                if Hc(jc).sys(js)==j
        h=h+c(k,jc)*Hc(jc).op{js};
                end
            end
       end
       gate=expm(-1i*dt*h);
       g1{j}=gate;
    end 
%forward propagation
for jt=1:nt
 %apply odd terms
for j=1:2:n-1
    [mps{j},mps{j+1}]=mps_gate_2q(mps{j},mps{j+1},g2{j},sv_min,D);
end
 %apply even terms
for j=2:2:n-1
    [mps{j},mps{j+1}]=mps_gate_2q(mps{j},mps{j+1},g2{j},sv_min,D);
end
%apply 1q terms
for j=1:n
    [mps{j}]=mps_gate_1q(mps{j},g1{j});
end
end
if mod(k-1,midstep)==0||k==nbin
    if iscpr==1
        mps=mps_compress(mps,sv_min,Dc,nsweep);
    end
mps=mps_normalize(mps);
end
for j=1:n
maxD=max(maxD,max(size(mps{j})));
end
end
ovl=mps_overlap(mpstg,mps);
iF=1-abs(ovl)^2;   
iFlist(jsampl)=iF;
Dlist(jsampl)=maxD;
end
if ismean==1
    iFlist=mean(iFlist);
    Dlist=mean(Dlist);
end
end