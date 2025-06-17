%apply tn0 to the left and tno+ to the right of {A0, B0}
%index order of A and B: 1,2,3,4 == left, bottom,top, right
%index order of tno: 1,2,3,4,5,6 == bottom right, bottom middle, bottom
%left, top right, top middle, top
%left, right=3rd qubit, middle=2nd qubit, left=1st qubit
function [A,B,C]=gate_3q_LR(A0,B0,C0,tno,sv_min,D) %apply the tensor network operator tno of a two adjacent qubit gate on the two tensors of the two qubits D is the truncation. tno has to be a 2x2x2x2 tensor
d=size(A0,2);
tno2=reshape(tno,[d^3,d^3]);
tno2=tno2';
tno2=reshape(tno2,[d,d,d,d,d,d]);
tno2=permute(tno2,[3,6,2,5,1,4]);
tno=permute(tno,[3,6,2,5,1,4]);
daL=size(A0,1);
dcR=size(C0,4);
%apply gate
T=tensorprod(A0,B0,4,1,NumDimensionsA=4);
T=tensorprod(T,C0,6,1,NumDimensionsA=6);
T=tensorprod(T,tno,[2,4,6],[2,4,6],NumDimensionsA=8);
T=permute(T,[1,6,2,7,3,8,4,5]);
T=tensorprod(T,tno2,[3,5,7],[1,3,5],NumDimensionsA=8);
T=permute(T,[1,2,6,3,7,4,8,5]);
nshape=[daL*d*d,d*d*d*d*dcR];
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
% Vhtemp=Vtemp';
A=reshape(Utemp,[daL,d,d,dtemp]);
O=reshape(S*Vhtemp,[dtemp,d,d,d,d,dcR]);
O=reshape(O,[dtemp*d*d,d*d*dcR]);
[Utemp,Stemp,Vtemp]=svd(O,'econ');
svlist=diag(Stemp);
svlist2=svlist.^2;
snorm=sum(svlist2);
normthres=sv_min*snorm;
dtemp2=length(svlist2);
normtest=svlist2(dtemp2);
while normtest<normthres
    dtemp2=dtemp2-1;
    normtest=normtest+svlist2(dtemp2);
end
dtemp2=min(dtemp2,D);
S=Stemp(1:dtemp2,1:dtemp2);%don't rescale this!
Utemp=Utemp(:,1:dtemp2);
Vhtemp=Vtemp(:,1:dtemp2)';
% Vhtemp=Vtemp';
B=reshape(Utemp,[dtemp,d,d,dtemp2]);
C=reshape(S*Vhtemp,[dtemp2,d,d,dcR]);
end