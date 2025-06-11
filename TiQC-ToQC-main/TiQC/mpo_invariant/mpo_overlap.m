function O=mpo_overlap(mpo,mpo0)%calculate tr(U+U0), inddices order of tns and tns0: 1:left, 2:down, 3:up, 4:right
%complex transpose of tns
n=length(mpo);
mpo=conjtp(mpo);
A=mpo0{1};
B=mpo{1};
O=tensorprod(A,B,[2,3],[3,2],NumDimensionsA=4);
O=permute(O,[1,3,2,4]);
for j=2:n
    A=mpo0{j};
    B=mpo{j};
    Otemp=tensorprod(A,B,[2,3],[3,2],NumDimensionsA=4);
    Otemp=permute(Otemp,[1,3,2,4]);
    O=tensorprod(O,Otemp,[3,4],[1,2],NumDimensionsA=4);
end
end