function [mpo2,new_norm,old_norm]=mpo_normalize(mpo1)
n=length(mpo1);
N1=mpo_overlap(mpo1,mpo1);
old_norm=N1;
mpo2=cell(1,n);
for j=1:n
    mpo2{j}=mpo1{j}/(sqrt(N1))^(1/n);
end
new_norm=mpo_overlap(mpo2,mpo2);
end