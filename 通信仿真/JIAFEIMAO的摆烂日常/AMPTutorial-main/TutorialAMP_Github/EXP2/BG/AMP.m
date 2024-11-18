function MSE_error=AMP_Detector(obj,Input)

IterNum=Input.IterNum;
N=Input.N;
M=Input.M;
H=obj.H;
y=obj.y;
mes=Input.mes;
MSE_error=zeros(IterNum,1);
nuw=Input.nuw;

%% array initialization
hat_v=ones(N,1);
hat_x=zeros(N,1);

sqrH=abs(H).^2;
sqrHt=sqrH';
Ht=H';
V_old=ones(M,1);
Z_old=y;

MSE_old=1;

for ii=1:IterNum
    
    %% Output Nodes
    V=sqrH*hat_v;
    Z=H*hat_x-V.*(y-Z_old)./(nuw+V_old);
    [Z,Z_old]=damping(Z,Z_old,mes);
    [V,V_old]=damping(V,V_old,mes);
      
    %% Input Nodes
    Sigma=(sqrHt*(1./(nuw+V))).^(-1);
    R=hat_x+Sigma.*(Ht*((y-Z)./(nuw+V)));  
    [hat_x,hat_v]=Estimator_x(Input, R, Sigma);
    MSE=norm(hat_x-obj.x)^2/norm(obj.x)^2;
    
    if sum(isnan(hat_v))>0 || MSE>MSE_old
       MSE_error(ii:IterNum,1)=MSE_old;
       break;
    end
    
    MSE_error(ii,1)=MSE;
    MSE_old=MSE;
      
end


end




function [hatx,varx]=Estimator_x(Input,r,Sigma)
rho=Input.rho;
sigma_X=1/rho;

Gau=@(x,a,v) 1./sqrt(2*pi*v).*exp(-1./(2*v).*abs(x-r).^2);

C=(rho*Gau(0,r,Sigma+sigma_X))./...
    (rho*Gau(0,r,Sigma+sigma_X)+(1-rho)*Gau(0,r,Sigma));

hatx=C.*r./(1+rho*Sigma);
Ex2=C.*(Sigma./(1+rho*Sigma)+(r./(1+rho*Sigma)).^2);
varx=Ex2-abs(hatx).^2;
end


