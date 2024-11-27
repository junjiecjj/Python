function   obj=system_model(Input)


N=Input.N;
M=Input.M;
nuw=Input.nuw;
rho=Input.rho;

%% Generate Bernoulli-Gaussian
x0 = sqrt(1/rho)*randn(N,1);               % a dense Gaussian vector
pos=rand(N,1) < rho;
x = x0.*pos;  

H=randn(M,N)/sqrt(M);
w=sqrt(nuw)*randn(M,1);   
y=H*x+w;

%% load parameters
obj.x=x;
obj.H=H;
obj.y=y;

end