function [MSE, error]=FISTA_Lasso(Input, obj)

% Load parameters
N=Input.N;
y=obj.y;
lambda=Input.lambda;
H=obj.H;


% Initialization 

t=0.2;   %step size

IterNum=Input.IterNum;

hatx_1=zeros(N,1);
hatx_2=zeros(N,1);
hatx_old=zeros(N,1);

mes=0.85;

for ii=1:IterNum
     
    hatx=hatx_1+(ii-2)/(ii+1)*(hatx_1-hatx_2);
    
    r = hatx - t*(H'*(H*hatx-y));
    hatx = sign(r).*( max( 0, abs(r)- lambda*t) );
    MSE(ii,1)=norm(hatx-obj.x)^2/norm(obj.x)^2;
    error(ii,1)=norm(y-H*hatx);
    
    [hatx, hatx_old]=damping(hatx, hatx_old, mes);
    
    hatx_2=hatx_1;
    hatx_1=hatx;
    
end




end


function [x, x_old]=damping(x, x_old, mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end