function [MSE, error] = AMP_Lasso(Input, obj)
    %% Load parameters
    IterNum=Input.IterNum;
    M=Input.M;
    N=Input.N;
    lambda=Input.lambda;
    y=obj.y;
    H=obj.H;
    hatx=zeros(N,1);
    delta=M/N;
    gamma=1;
    Onsager=0;
    mes=0.95;
    z_old=0;

    for ii=1:IterNum
        z=y-H*hatx+Onsager;
        [z, z_old]=damping(z, z_old ,mes);
        r=hatx+H'*z;
        hatx=sign(r).*max(abs(r)-lambda-gamma,0);
        tem=mean_partial(hatx+H'*z, lambda+gamma);
        Onsager=1/delta*z*mean_partial(hatx+H'*z, lambda+gamma);   
        gamma=(lambda+gamma)/delta*mean_partial(hatx+H'*z, lambda+gamma);
        
        MSE(ii,1)=norm(obj.x-hatx)^2/norm(obj.x)^2;
        error(ii,1)=norm(y-H*hatx);
    end
end

function mean_out=mean_partial(R, Sigma)
    N=length(R);
    tem=zeros(N,1);
    for ii=1:N
        if abs(R(ii))>Sigma
            tem(ii)=1;
        end
    end
    mean_out=mean(tem);
end

function [x, x_old]=damping(x, x_old, mes)
    x=mes*x+(1-mes)*x_old;
    x_old=x;
end
