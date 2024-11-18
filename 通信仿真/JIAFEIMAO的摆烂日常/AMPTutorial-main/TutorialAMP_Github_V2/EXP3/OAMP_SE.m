function MSE_array =OAMP_SE(Input,obj)

N=Input.N;
M=Input.M;
IterNum=Input.IterNum ;
nuw=Input.nuw;
rho=Input.rho;
lambda=obj.lambda;

Gaussian=@(x,m,v) 1./sqrt(2*pi*v).*exp(-(x-m).^2./(2*v));

hatv=1;

Lambda=lambda.^2;
for ii=1:IterNum 

Sigma_x=hatv*((sum(Lambda./(Lambda+nuw/hatv))/N)^(-1)-1);

MSE=1-rho/(1+rho*Sigma_x)*integral(@(z) z.^2./(rho+(1+rho)*sqrt((1+rho*Sigma_x)/(rho*Sigma_x))*exp(-z.^2./(2*rho*Sigma_x))+eps).*Gaussian(z,0,1),-Inf, Inf);
MSE=real(MSE);
MSE_array(ii)=MSE;

hatv=max(1/(1/MSE-1/Sigma_x), 1e-6);
end

end