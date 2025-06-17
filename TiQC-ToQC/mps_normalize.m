function [mps2,new_norm,old_norm]=mps_normalize(mps1)
n=length(mps1);
N1=mps_overlap(mps1,mps1);
old_norm=N1;
mps2=cell(1,n);
for j=1:n
    mps2{j}=mps1{j}/(sqrt(N1))^(1/n);
end
new_norm=mps_overlap(mps2,mps2);
end