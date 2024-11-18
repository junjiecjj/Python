function MSE_array =AMP_SE(Input)

N=Input.N;
M=Input.M;
IterNum=Input.IterNum ;
alpha=M/N;
nuw=Input.nuw;
rho=Input.rho;

Gaussian=@(x,m,v) 1./sqrt(2*pi*v).*exp(-(x-m).^2./(2*v));

MSE=1;

for ii=1:IterNum 
Sigma_x=nuw+1/alpha*MSE;

MSE=1-rho/(1+rho*Sigma_x)*integral(@(z) z.^2./(rho+(1+rho)*sqrt((1+rho*Sigma_x)/(rho*Sigma_x))*exp(-z.^2./(2*rho*Sigma_x))+eps).*Gaussian(z,0,1),-Inf, Inf);
 
MSE_array(ii)=MSE;
end

end