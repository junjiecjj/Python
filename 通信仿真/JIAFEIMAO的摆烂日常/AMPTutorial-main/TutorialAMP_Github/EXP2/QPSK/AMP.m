function MSE_error=AMP(obj,Input)

IterNum=Input.IterNum;
N=Input.N;
M=Input.M;
H=obj.H;
xo=obj.xo;
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
Z_old=zeros(M,1);

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
    [hat_x,hat_v]=Estimator_x(xo,R, Sigma);
    MSE=norm(hat_x-obj.x)^2/norm(obj.x)^2;
    
    if sum(isnan(hat_v))>0 || MSE>MSE_old
       MSE_error(ii:IterNum,1)=MSE_old;
       break;
    end
    
    MSE_error(ii,1)=MSE;
    MSE_old=MSE;
      
end


end




function [hatx,varx]=Estimator_x(xo, r, Sigma)
log_posterior=bsxfun(@times,-1./Sigma,abs(bsxfun(@minus,xo,r).^2));
log_posterior=bsxfun(@minus,log_posterior,max(log_posterior));  %防止溢出

posterior=exp(log_posterior); 
posterior=bsxfun(@rdivide,posterior,sum(posterior,2));       %得到标准PDF
hatx=sum(bsxfun(@times,posterior,xo),2);                     %计算PDF的均值
varx=sum(posterior.*abs(bsxfun(@minus,hatx,xo).^2),2);       %计算PDF的方差
end


