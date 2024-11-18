function   obj=MIMO_system(Input)

N=Input.N;
M=Input.M;
nuw=Input.nuw;
kappa=Input.kappa;
rho=Input.rho;

%% Generate x
x0 = sqrt(1/rho)*randn(N,1);               % a dense Gaussian vector
pos=rand(N,1) < rho;
x = x0.*pos;  % insert zeros


%% Channel
U=orth(randn(M));
V=orth(randn(N));

if kappa==1
    lambda=ones(M,1);
else 
    lambda=logspace(log10(1),log10(kappa),M);
end
lambda=lambda/sqrt(norm(lambda)^2/N);
Sigma=diag(lambda);
Sigma=[Sigma, zeros(M,N-M)];
H=U*Sigma*V';



%% Noise
w=sqrt(nuw)*randn(M,1);   %产生高斯噪声

%% Uncoded system
y=H*x+w;

%% load parameters
obj.x=x;
obj.y=y;
obj.H=H;

end