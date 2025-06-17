function mpo1=conjtp(mpo)%calculate conjugate transpose of a tnoperator
n=length(mpo);
%complex transpose of tns
    for j=1:n
    mpo{j}=permute(conj(mpo{j}),[1,3,2,4]);
    end
mpo1=mpo;
end