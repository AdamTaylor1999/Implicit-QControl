function I=mpo_to_full(mpo)
n=length(mpo);
A=mpo{1};
B=mpo{2};
tensors={A,B};
connects={[-1,-2,-4,1],[1,-3,-5,-6]};
I=ncon(tensors,connects);
for j=3:n
    B=mpo{j};
    tensors={I,B};
    connects={[-1,-2:-1:-j,-(j+2):-1:-2*j,1],[1,-(j+1),-(2*j+1),-(2*j+2)]};
    I=ncon(tensors,connects);
end
I=reshape(I,[2^n,2^n]);
end