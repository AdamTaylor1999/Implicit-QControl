function [iF,iG]=mps_infid_T_robust_prl(J,H0,Hu,Hc,x,mps0,mpstg,varT,ismean,tebd_options)
T=x(end);
c=x(1:end-1);
[iF,iG1]=mps_infid_robust_prl(J,H0,Hu,Hc,c,T,mps0,mpstg,ismean,tebd_options);
if varT==1
dt=10^(-10);    
T=T+dt;
iF2=mps_infid_nograd_robust_prl(J,H0,Hu,Hc,c,T,mps0,mpstg,ismean,tebd_options);
iGT=(iF2-iF)/dt;
else
iGT=0;
end
iG=[iG1;iGT];
end
