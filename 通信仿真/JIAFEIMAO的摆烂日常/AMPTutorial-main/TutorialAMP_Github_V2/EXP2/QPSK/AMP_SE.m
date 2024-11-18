function MSE_array =AMP_SE(Input)

N=Input.N;
M=Input.M;
IterNum=Input.IterNum ;
alpha=M/N;
nuw=Input.nuw;

Gaussian=@(x,m,v) 1./sqrt(2*pi*v).*exp(-(x-m).^2./(2*v));

MSE=1;

for ii=1:IterNum 
tau=nuw+1/alpha*MSE;
gamma=1/tau;
MSE=1-integral(@(u) tanh(gamma+sqrt(gamma).*u).*Gaussian(u,0,1),-Inf,Inf);
MSE_array(ii)=MSE;
end

end