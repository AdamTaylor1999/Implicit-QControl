function mps1=mps_compress(mps,sv_min,Dc,nsweep)
n=length(mps);
mps1=mps;
for jsweep=1:nsweep
    %left sweep
for j=1:n-1
A=mps1{j};
B=mps1{j+1};
dA=size(A);
dB=size(B);
if length(dA)==2
    dA(3)=1;
end
if length(dB)==2
    dB(3)=1;
end
MA=reshape(A,[dA(1)*dA(2),dA(3)]);
MB=reshape(B,[dB(1),dB(2)*dB(3)]);
[Utemp,Stemp,Vtemp]=svd(MA,'econ');
svlist=diag(Stemp);
dtemp=length(svlist);
S=Stemp(1:dtemp,1:dtemp);
S=S/norm(S);
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
 mps1{j}=reshape(Utemp,[dA(1),dA(2),dtemp]);
 mps1{j+1}=reshape(S*Vhtemp*MB,[dtemp,dB(2),dB(3)]);
end
%right sweep
for j=n:-1:2
A=mps1{j};
B=mps1{j-1};
dA=size(A);
dB=size(B);
if length(dA)==2
    dA(3)=1;
end
if length(dB)==2
    dB(3)=1;
end
MA=reshape(A,[dA(1),dA(2)*dA(3)]);
MB=reshape(B,[dB(1)*dB(2),dB(3)]);
[Utemp,Stemp,Vtemp]=svd(MA,'econ');
svlist=diag(Stemp);
svlist2=svlist.^2;
snorm=sum(svlist2);
norm_thres=sv_min*snorm;
dtemp=length(svlist2);
normtest=svlist2(dtemp);
while normtest<norm_thres
    dtemp=dtemp-1;
    normtest=normtest+svlist2(dtemp);
end
dtemp=min(dtemp,Dc);
S=Stemp(1:dtemp,1:dtemp);
S=S/sqrt(snorm);
% for jj=1:length(svlist2)
% normlist(jj)=sum(svlist2(jj:end));
% end
% dtemp=find(normlist>sv_min*normlist(1),1,'last');
Utemp=Utemp(:,1:dtemp);
Vhtemp=Vtemp(:,1:dtemp)';
 mps1{j}=reshape(Vhtemp,[dtemp,dA(2),dA(3)]);
 mps1{j-1}=reshape(MB*Utemp*S,[dB(1),dB(2),dtemp]);
end
end