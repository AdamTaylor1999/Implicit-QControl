function mpo=mpo_add(mpo1,mpo2)
n=size(mpo1,2);
j=1;
d1=size(mpo1{j});
d2=size(mpo2{j});
if length(d1)==3
    d1(4)=1;
end
if length(d2)==3
    d2(4)=1;
end
W=[reshape(mpo1{j},[d1(1)*d1(2),d1(3)*d1(4)]),reshape(mpo2{j},[d2(1)*d2(2),d2(3)*d2(4)])];
W=reshape(W,[1,2,2,d1(4)+d2(4)]);
mpo{j}=W;
for j=2:n-1
d1=size(mpo1{j});
d2=size(mpo2{j});
if length(d1)==3
    d1(4)=1;
end
if length(d2)==3
    d2(4)=1;
end
temp1=permute(mpo1{j},[2,1,3,4]);
temp2=permute(mpo2{j},[2,1,3,4]);
W=blkdiag(reshape(temp1,[d1(1)*d1(2),d1(3)*d1(4)]),reshape(temp2,[d2(1)*d2(2),d2(3)*d2(4)]));
W=reshape(W,[2,d1(1)+d2(1),2,d1(4)+d2(4)]);
W=permute(W,[2,1,3,4]);
mpo{j}=W;
end
j=n;
d1=size(mpo1{j});
d2=size(mpo2{j});
if length(d1)==3
    d1(4)=1;
end
if length(d2)==3
    d2(4)=1;
end
temp1=permute(mpo1{j},[2,1,3,4]);
temp2=permute(mpo2{j},[2,1,3,4]);
W=[reshape(temp1,[d1(1)*d1(2),d1(3)*d1(4)]);reshape(temp2,[d2(1)*d2(2),d2(3)*d2(4)])];
W=reshape(W,[2,d1(1)+d2(1),2,1]);
W=permute(W,[2,1,3,4]); 
mpo{j}=W;
end