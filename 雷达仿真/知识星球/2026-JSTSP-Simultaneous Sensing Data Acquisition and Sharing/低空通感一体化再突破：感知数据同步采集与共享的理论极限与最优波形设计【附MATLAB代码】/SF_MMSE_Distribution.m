%% 此程序解决信号模型Z=XS+N中,X和S是随机变量，X是信道输入，S是待估计变量，N是随机噪声
%% 在给定X时, 利用接收信号Z对变量S做MMSE估计，得到估计量hat(S)
%% 那么，在给定X和S的分布时，hat(S)的分布是什么？
%% 该程序是求取hat(S)的数值分布
%% By Fuwang Dong 2024/05/03

function   P_hatS   =  SF_MMSE_Distribution(P_X,X_input,hatS,sigma_noise)

%%   对于给定的X，模型Z=XS+N的MMSE估计为hat(S)=X/(1+X^2)*Z=X/(1+X^2)*(XS+N)
%%   这里我们考虑Gaussian信道，即噪声N服从CN(0,1)
%%   对于信源分布S，先考虑高斯信源的情况，即S服从CN(0,1)高斯分布
%%  计算P(hatS|X),从表达式上看，hatS为两个Guassian分布的和，其分布仍为Guassian
%%  hatS=X^2/(1+X^2)*S + X/(1+X^2)*N, 均值为0，方差为X^2/(1+X^2)

N_symbol               =  length(X_input);
P_hatS_given_X         =  zeros(N_symbol,N_symbol); 
sigma2                 =  X_input.^2./(sigma_noise+X_input.^2);          %给定X时p(hats|x)的方差
sigma2(sigma2==0)      =  1e-6;                                          %去掉nan的部分

%% 计算给定X时P(hatS|X)
for  iter  =  1:N_symbol
    sigma2_given_X             =  sigma2(iter);
    P_temp                     =  1./sqrt(2*pi*sigma2_given_X)*exp(-hatS.^2./(2*sigma2_given_X));   
    P_hatS_given_X(:,iter)     =  P_temp/sum(P_temp);                    %hatS仍为Gaussian分布,且归一化PDF
end

%% 计算P(hatS)=P(hatS|X)P(X)
P_hatS                 =  P_hatS_given_X*P_X;
end



  
