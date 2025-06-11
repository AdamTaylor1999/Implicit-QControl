function A=mps_gate_1q(A0,tno) %apply the tensor network operator tno of a single qubit gate on single-qubit tensor A0
A=tensorprod(A0,tno,2,2,NumDimensionsA=3);
A=permute(A,[1,3,2]);
end