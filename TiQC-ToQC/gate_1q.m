%index order of A0: 1,2,3,4 == left, bottom,top, right
function A=gate_1q(A0,tno) %apply the tensor network operator tno of a single qubit gate on single-qubit tensor A0
A=tensorprod(A0,tno,2,2,NumDimensionsA=4);
A=permute(A,[1,4,2,3]);
end