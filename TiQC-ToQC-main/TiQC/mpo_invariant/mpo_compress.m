function mpo1=mpo_compress(mpo,sv_min,D,nsweep)
n=length(mpo);
mpo1=mpo;
for jsweep=1:nsweep
    %left sweep
for j=1:n-1
A=mpo1{j};
B=mpo1{j+1};
dA=size(A);
dB=size(B);
if length(dA)==3
    dA(4)=1;
end
if length(dB)==3
    dB(4)=1;
end
MA=reshape(A,[dA(1)*dA(2)*dA(3),dA(4)]);
MB=reshape(B,[dB(1),dB(2)*dB(3)*dB(4)]);
[Utemp,Stemp,Vtemp]=svd(MA,'econ');
svlist=diag(Stemp);
dtemp=length(svlist);
S=Stemp(1:dtemp,1:dtemp);
S=S/norm(S);
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
 mpo1{j}=reshape(Utemp,[dA(1),dA(2),dA(3),dtemp]);
 mpo1{j+1}=reshape(S*Vhtemp*MB,[dtemp,dB(2),dB(3),dB(4)]);
end
%right sweep
for j=n:-1:2
A=mpo1{j};
B=mpo1{j-1};
dA=size(A);
dB=size(B);
if length(dA)==3
    dA(4)=1;
end
if length(dB)==3
    dB(4)=1;
end
MA=reshape(A,[dA(1),dA(2)*dA(3)*dA(4)]);
MB=reshape(B,[dB(1)*dB(2)*dB(3),dB(4)]);
[Utemp,Stemp,Vtemp]=svd(MA,'econ');
svlist=diag(Stemp);
svlist2=svlist.^2;
snorm=sum(svlist2);
norm_thres=sv_min*snorm;
dtemp=length(svlist2);
normtest=svlist2(dtemp);
while normtest<norm_thres
    dtemp=dtemp-1;
    normtest=normtest+svlist2(dtemp);
end
dtemp=min(dtemp,D);
S=Stemp(1:dtemp,1:dtemp);
S=S/sqrt(snorm);
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
 mpo1{j}=reshape(Vhtemp,[dtemp,dA(2),dA(3),dA(4)]);
 mpo1{j-1}=reshape(MB*Utemp*S,[dB(1),dB(2),dB(3),dtemp]);
end
end