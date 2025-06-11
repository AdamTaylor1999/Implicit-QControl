function [mpo,Dmax]=full_to_mpo(I)
n=round(log2(length(I)));
mpo=cell(n,1);
dim_list=2*ones(1,2*n);
I=reshape(I,dim_list);
perm_list=zeros(1,2*n);
perm_list(1:2:2*n)=n:-1:1;
perm_list(2:2:2*n)=2*n:-1:n+1;
I=permute(I,perm_list);
eta=1;
Dmax=1;
for j=1:n-1
I=reshape(I,[eta*2^2,4^(n-j)]);
[U,S,V]=svd(I,'econ'); 
Vh=V';
beta=size(S,1);
U=U*sqrt(S); 
Vh=sqrt(S)*Vh;
mpo{j}=reshape(U,[eta,2,2,beta]);
Dmax=max(Dmax,max(size(mpo{j})));
I=Vh;
eta=beta;
end
mpo{n}=reshape(Vh,[eta,2,2,1]);
end