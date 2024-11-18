function [MSE, error]=ISTA_Lasso(Input, obj)

% Load parameters
N=Input.N;
y=obj.y;
lambda=Input.lambda;
H=obj.H;

% Initialization 
hatx=zeros(N,1);
t=0.35;   %step size

IterNum=Input.IterNum;

for ii=1:IterNum
        
    r = hatx - t*(H'*(H*hatx-y));
    hatx = sign(r).*( max( 0, abs(r)- lambda*t) );
    MSE(ii,1)=norm(hatx-obj.x)^2/norm(obj.x)^2;
    error(ii,1)=norm(y-H*hatx);
    
end




end