function  MSE_array=MAMP_Detector(obj,Input)

%% load parameters
H=obj.H;
y=obj.y;
N=Input.N;
M=Input.M;
x=obj.x;
alpha=M/N;
nuw=Input.nuw;
IterNum=Input.IterNum;
lambda=obj.lambda.^2;
mes1=0.7;        %damping factor of rz and hax
mes2=0.7;        %damping factor of hat_v


HHt=H*H';
lambda_dag=(max(lambda)+min(lambda))/2;
B=lambda_dag*eye(M)-HHt;


[w, w0, over_w, over_w00,  over_wt0]=Cal_wn(Input, lambda, lambda_dag, IterNum);



%% parameter initialization
z=zeros(M,1);
hatx_array=zeros(N,IterNum+1);
hatv_array(1,1)=(1/N*(y'*y)-alpha*nuw)/w0;
rz=y;                                         %residual error rz=y-Hhat_x;
rz_array(:,1)=rz;

MSE_old=1;
xi(1)=1;

vartheta(1,1)=1;
hatx_old=zeros(N,1);
rz_old=y;
%% Iteration 
for t=1:IterNum
    
    rho=nuw/hatv_array(t,t);
    theta=(lambda_dag+rho)^(-1);
    
    
    %% Get p and vartheta 
    p_bar=zeros(t,1); 
    if t>1
        vartheta_p(1:t-1,1)=vartheta(1:t-1,1);
        for ii=1:t-2
            vartheta(ii,1)=theta*vartheta_p(ii,1);
            p_bar(ii,1)=vartheta(ii,1)*w(t-ii);
        end
        vartheta(t-1,1)=xi(t-1)*theta;
        p_bar(t-1,1)=vartheta(t-1,1)*w(1);     
    end

    %% Get c
    c1=nuw*w0+hatv_array(t,t)*over_w00; 
    if t==1
       [c0, c2, c3]=deal(0);
    else
         c0=sum(p_bar(1:t-1)/w0); 
         c2=0;  c3=0;
         for ii=1:t-1
            c2=c2-vartheta(ii,1)*(nuw*w(t-ii)+hatv_array(t,ii)* over_wt0(1,t-ii));
            for jj=1:t-1
                c3=c3+vartheta(ii,1)*vartheta(jj,1)*(nuw*w(2*t-ii-jj)+hatv_array(ii,jj)*over_w(t-ii,t-jj));
            end
         end
        temp=c1*c0+c2;
        if temp ~=0
            xi(t)=(c2*c0+c3)/temp;
        else
            xi(t)=1;
        end
        vartheta(t,1)=xi(t);
    end
   
    p_bar(t,1)=xi(t)*w0;
    vareps=p_bar(t)+w0*c0;
    tau=(c1*xi(t)^2-2*c2*xi(t)+c3)/vareps^2;
    z=theta*B*z+xi(t)*rz;
    
    tem_hatx=hatx_array(:,1:t)*p_bar(1:t,1);
    r=1/vareps*(H'*z+tem_hatx);
    
    % MMSE estimator
    [hatx_mmse,varx]=EstimatorX(Input,r,tau);

    MSE=norm(x-hatx_mmse)^2/norm(x)^2;
    MSE_array(t,1)=MSE;
    
    if isnan(varx)>0 || isinf(varx)>0
        MSE_array(t:IterNum,1)=MSE_old;
        break
    end
    MSE_old=MSE;
    
    hatv=1/(1/varx-1/tau);
    hatx=hatv*(hatx_mmse/varx-r/tau);   

    rz=y-H*hatx;
    rz_array(:,t+1)=rz;
    for jj=1:t+1
        hatv_array(t+1,jj)=(1/N*rz'*rz_array(:,jj)-alpha*nuw)/w0;
        hatv_array(jj,t+1)=hatv_array(t+1,jj);
    end


   [hatx, hatx_old]=damping(hatx, hatx_old, mes1);
   [rz, rz_old]=damping(rz, rz_old, mes1);
   hatx_array(:,t+1)=hatx;
   
   hatv_array=DampingV(hatv_array, mes2, t);

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
varx=mean(Ex2-abs(hatx).^2);
end


function [x,x_old]=damping(x, x_old, mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end

function hatv_array=DampingV(hatv_array, mes, t)

for ii=1:t+1
    hatv_array(t+1,ii)=mes*hatv_array(t+1,ii)+(1-mes)*hatv_array(t,ii);
    hatv_array(ii,t+1)=hatv_array(t+1,ii);
end


end


