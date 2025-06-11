function O=mps_overlap(mps,mps0)
N=length(mps0);
A=mps0{1};
B=conj(mps{1});
O=tensorprod(A,B,2,2,NumDimensionsA=3);
O=permute(O,[1,3,2,4]);
for j=2:N
    A=mps0{j};
    B=conj(mps{j});
    Otemp=tensorprod(A,B,2,2,NumDimensionsA=3);
    Otemp=permute(Otemp,[1,3,2,4]);
    O=tensorprod(O,Otemp,[3,4],[1,2],NumDimensionsA=4);
end
%beware of edge cases where dimension of tensor Otemp is only 2
end