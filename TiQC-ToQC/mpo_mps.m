function mps=mpo_mps(mpo,mps0)%apply mpo on mps: 1:left, 2:down, 3:up, 4:right
n=length(mpo);
mps=mps0;
A=mpo{1};
B=mps0{1};%this is conjugation, not complex transpose
O=tensorprod(A,B,3,2,NumDimensionsA=4);
O=permute(O,[1,4,2,3,5]);
O=reshape(O,[size(A,1)*size(B,1),size(A,2),size(A,4),size(B,3)]);
for j=2:n
    A=mpo{j};
    B=mps0{j};
    Otemp=tensorprod(A,B,3,2,NumDimensionsA=4);
    Otemp=permute(Otemp,[1,4,2,3,5]);  
    O=tensorprod(O,Otemp,[3,4],[1,2],NumDimensionsA=4);
    T=reshape(O,[size(O,1)*size(O,2),size(A,2)*size(A,4)*size(B,3)]);
    [Utemp,Stemp,Vtemp]=svd(T,'econ');
    Vhtemp=Vtemp';
    dtemp=length(Stemp);
    mps{j-1}=reshape(Utemp*sqrt(Stemp),[size(O,1),size(O,2),dtemp]);
    O=reshape(sqrt(Stemp)*Vhtemp,[dtemp,size(A,2),size(A,4),size(B,3)]);
end
mps{n}=reshape(O,[size(O,1),size(O,2),size(O,3)*size(O,4)]);
end