function [A,B]=mps_gate_2q(A0,B0,tno,sv_min,D) %apply the tensor network operator tno of a two adjacent qubit gate on the two tensors of the two qubits D is the truncation. tno has to be a 2x2x2x2 tensor
size_temp=size(A0);
d=size_temp(2);
dA0=size(A0,1);
dB0=size(B0,3);
%apply gate
Ttemp=tensorprod(A0,B0,3,1,NumDimensionsA=3);
tno=permute(tno,[2,1,4,3]);
T=tensorprod(Ttemp,tno,[2,3],[3,4],NumDimensionsA=4);
T=permute(T,[1,3,4,2]);
nshape=[d*dA0,d*dB0];
T=reshape(T,nshape);
%truncation
[Utemp,Stemp,Vtemp]=svd(T,'econ');
svlist=diag(Stemp);
svlist2=svlist.^2;
snorm=sum(svlist2);
normthres=sv_min*snorm;
dtemp=length(svlist2);
normtest=svlist2(dtemp);
while normtest<normthres
    dtemp=dtemp-1;
    normtest=normtest+svlist2(dtemp);
end
dtemp=min(dtemp,D);
S=Stemp(1:dtemp,1:dtemp);
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
A=reshape(Utemp*sqrt(S),[dA0,d,dtemp]);
B=reshape(sqrt(S)*Vhtemp,[dtemp,d,dB0]);
end