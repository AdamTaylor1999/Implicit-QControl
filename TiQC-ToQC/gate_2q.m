%index order of A and B: 1,2,3,4 == left, bottom,top, right
%index order of tno: 1,2,3,4 == bottom right, bottom left, top right, top
%left, right=2nd qubit, left=1st qubit
function [A,B]=gate_2q(A0,B0,tno,sv_min,D) %apply the tensor network operator tno of a two adjacent qubit gate on the two tensors of the two qubits D is the truncation. tno has to be a 2x2x2x2 tensor
size_temp=size(A0);
d=size_temp(2);
dA0=size(A0,1);
dB0=size(B0,4);
%apply gate
Ttemp=tensorprod(A0,B0,4,1,NumDimensionsA=4);
tno=permute(tno,[2,4,1,3]);


T=tensorprod(Ttemp,tno,[2,4],[2,4],NumDimensionsA=6);


T=permute(T,[1,5,2,6,3,4]);
nshape=[d*d*dA0,d*d*dB0];
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
S=Stemp(1:dtemp,1:dtemp);%don't rescale this!
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
A=reshape(Utemp*sqrt(S),[dA0,d,d,dtemp]);
B=reshape(sqrt(S)*Vhtemp,[dtemp,d,d,dB0]);
end