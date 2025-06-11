%index order of A0: 1,2,3,4 == left, bottom,top, right
function A=gate_1q_LR(A0,U) %apply the operator U of a single qubit gate on single-qubit tensor A0
Uc=U';
A1=tensorprod(A0,U,2,2,NumDimensionsA=4);
A1=permute(A1,[1,4,2,3]);
A=tensorprod(A1,Uc,3,1,NumDimensionsA=4);
A=permute(A,[1,2,4,3]);
end