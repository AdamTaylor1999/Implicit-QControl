function [mpo,ovl,Dlist]=mpo_evol_3q(H0,H1,Hc,c0,c1,c,T,mpo0,tebd_options)
sv_min=tebd_options.sv_min;
D=tebd_options.bond_dim;
Dc=tebd_options.bond_comp;
nsweep=tebd_options.num_sweep;
midstep=tebd_options.num_midstep;
nt=tebd_options.num_refined_step;
iscpr=tebd_options.is_compressed;
n=length(mpo0);
nc=length(Hc);
nbin=length(c)/(nc);
c=reshape(c,[nbin,nc]);
m0=size(H0,2);
c0=reshape(c0,[nbin,m0]);
m1=size(H1,2);
c1=reshape(c1,[nbin,m1]);
d=size(mpo0{1},2);
dt=T/(nbin*nt);
ovl=zeros(nbin,1);
mpo=mpo0;
Dlist=zeros(ceil(nbin/midstep)+2,1);
jD=0;
g2=cell(1,n-1);
g3=cell(1,n-2);
for k=1:nbin
    %if mod(k-1,100)==0
        %fprintf('step %d\n',k);
    %end
     %gate construction
   for j=1:n-1
       h=0;
       for l=1:m0
        h=h+H0{j,l}*c0(k,l);
       end
        gate=expm(-1i*dt*h);
        gate=reshape(gate,[d,d,d,d]);
        g2{j}=gate;
   end
  for j=1:n-2
       h=0;
       for l=1:m1
        h=h+H1{j,l}*c1(k,l);
       end
        gate=expm(-1i*dt*h);
        gate=reshape(gate,[d,d,d,d,d,d]);
        g3{j}=gate;
   end
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
    [mpo{j},mpo{j+1}]=gate_2q_LR(mpo{j},mpo{j+1},g2{j},sv_min,D);
end
 %apply even terms
for j=2:2:n-1
    [mpo{j},mpo{j+1}]=gate_2q_LR(mpo{j},mpo{j+1},g2{j},sv_min,D);
end
%3q
%apply 1st terms
for j=1:3:n-2
    [mpo{j},mpo{j+1},mpo{j+2}]=gate_3q_LR(mpo{j},mpo{j+1},mpo{j+2},g3{j},sv_min,D);
end
 %apply 2nd terms
for j=2:3:n-2
     [mpo{j},mpo{j+1},mpo{j+2}]=gate_3q_LR(mpo{j},mpo{j+1},mpo{j+2},g3{j},sv_min,D);
end
 %apply 3rd terms
for j=3:3:n-2
    [mpo{j},mpo{j+1},mpo{j+2}]=gate_3q_LR(mpo{j},mpo{j+1},mpo{j+2},g3{j},sv_min,D);
end
%apply 1q terms
for j=1:n
    [mpo{j}]=gate_1q_LR(mpo{j},g1{j});
end
end
if iscpr==1
mpo=mpo_compress(mpo,sv_min,Dc,nsweep);
end
mpo=mpo_normalize(mpo);
jD=jD+1;
maxD=1;
for j=1:n
maxD=max(maxD,max(size(mpo{j})));
end
Dlist(jD)=maxD;
ovl(k)=mpo_overlap(mpo0,mpo);
end
end   
