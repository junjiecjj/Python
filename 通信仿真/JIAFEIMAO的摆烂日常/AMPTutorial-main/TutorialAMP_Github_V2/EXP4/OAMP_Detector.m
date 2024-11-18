function  MSE=OAMP_Detector(obj,Input)

% load parameters
H=obj.H;
y=obj.y;
N=Input.N;
x=obj.x;
nuw=Input.nuw;
mes=Input.mes;
IterNum=Input.IterNum;

r2=zeros(N,1);
r2_old=r2;
v2_inv=1;
v2_inv_old=v2_inv;

MSE=zeros(IterNum,1);

for ii=1:IterNum

    
    % Back Passing
    hatW=((H'*H)+nuw*v2_inv*eye(N))^(-1)*H';
    v1=(N/trace(hatW*H)-1)/v2_inv;
    v1=real(v1);
    r1=r2+N/trace(hatW*H)*hatW*(y-H*r2);
    
    
    %Forward Passing
    [hatx1,varx1]=EstimatorX(Input,r1,v1); 
    MSE(ii,1)=norm(hatx1-x).^2/norm(x).^2;
    varx1=mean(varx1);
    varx1=max(varx1,5e-13);
    v2_inv=(v1-varx1)/varx1/v1;
    r2=((hatx1*v1-r1*varx1)/varx1/v1)/v2_inv;
    
    if v2_inv<0
       v2_inv=v2_inv_old;
       r2=r2_old;
    end
    
    [v2_inv, v2_inv_old]=damping(v2_inv, v2_inv_old, mes);
    [r2, r2_old]=damping(r2, r2_old, mes);
end
end

function [hatx,varx]=EstimatorX(Input,r,Sigma)
rho=Input.rho;
sigma_X=1/rho;

Gau=@(x,a,v) 1./sqrt(2*pi*v).*exp(-1./(2*v).*abs(x-r).^2);

C=(rho*Gau(0,r,Sigma+sigma_X))./...
    (rho*Gau(0,r,Sigma+sigma_X)+(1-rho)*Gau(0,r,Sigma));

hatx=C.*r./(1+rho*Sigma);
Ex2=C.*(Sigma./(1+rho*Sigma)+(r./(1+rho*Sigma)).^2);
varx=Ex2-abs(hatx).^2;
end


function [x,x_old]=damping(x, x_old, mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end