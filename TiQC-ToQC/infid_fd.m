function [iF,iG] = infid_fd(H0,Hc,x,mpo0,mpotg,varT,tebd_options)
iF=infid_nograd(H0,Hc,x,mpo0,mpotg,tebd_options);
if nargout>1
dx=10^(-10);
iG=zeros(length(x),1);
if varT==1
parfor jx=1:length(x)
    x1=x;
    x1(jx)=x1(jx)+dx;
    iF1=infid_nograd(H0,Hc,x1,mpo0,mpotg,tebd_options);
    iG(jx)=(iF1-iF)/dx;
end
else
parfor jx=1:length(x)-1
    x1=x;
    x1(jx)=x1(jx)+dx;
    iF1=infid_nograd(H0,Hc,x1,mpo0,mpotg,tebd_options);
    iG(jx)=(iF1-iF)/dx;
end
end
end
end