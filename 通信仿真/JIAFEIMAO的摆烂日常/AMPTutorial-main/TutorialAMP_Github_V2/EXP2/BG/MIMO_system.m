function   obj=MIMO_system(Input)

N=Input.N;
M=Input.M;
nuw=Input.nuw;
rho=Input.rho;

%% Generate xo
x0 = sqrt(1/rho)*randn(N,1);               % a dense Gaussian vector
pos=rand(N,1) < rho;
x = x0.*pos;  % insert zeros

%% Channel
H=randn(M,N)/sqrt(M);

%% Noise
w=sqrt(nuw)*randn(M,1);   %产生高斯噪声



%% Uncoded system
y=H*x+w;

%% load parameters
obj.x=x;
obj.H=H;
obj.y=y;


end