% 本文复现Q Shi et al的经典论文《An iteratively weighted MMSE approach to distributed sum-utility maximization for a MIMO interfering broadcast channel》，
% https://zhuanlan.zhihu.com/p/586660620
% https://zhuanlan.zhihu.com/p/588715673


clear;
clc;
close all;


clc;clear;
K = 4; % 基站个数
T = 3; % 发射天线个数
R = 2; % 接收天线个数
epsilon = 1e-3; % 收敛条件
sigma2 = 1; % 噪声功率
snr = 25; % 信噪比
P = db2pow(snr)*sigma2; % 发射功率

I = 2; % 每个基站服务的用户个数
alpha1 = ones(I,K); % 权重系数，都假设相同

d = R; % 假设每个用户有R路独立的数据流

max_iter = 100;

%产生信道
H = cell(I,K,K); % 信道系数
for i=1:I
    for k = 1:K
        for j=1:K
           H{i,k,j}=sqrt(1/2)*(randn(R,T)+1i*randn(R,T));
        end
    end
end


%WMMSE算法
rate = []; % 初始化一个空向量记录rate

U = cell(I,K);
U(:)={zeros(R,d)};

% 随机初始化发射波束向量
V = cell(I,K); % 算法第一行
for i=1:I
    for k=1:K
        v = randn(T,d)+1i*randn(T,d); % 随机初始化
        V{i,k}=sqrt(P/I)*v/norm(v,"fro");
    end
end 

% 求初始化发射波束V后求系统和速率
rate_old = sum_rate(H,V,sigma2,R,I,K,alpha1);
rate = [rate rate_old];

iter1 = 1;
while(1)
    U = find_U(H,V,sigma2,R,I,K,d); % Tbale I line 4 in p.4435
    W = find_W(U,H,V,I,K,d); % Tbale I line 5 in p.4435
    V = find_V(alpha1,H,U,W,T,I,K,P); % Tbale I line 6 in p.4435
    rate_new = sum_rate(H,V,sigma2,R,I,K,alpha1);
    rate = [rate rate_new];
    iter1 = iter1 + 1;
    if abs(rate_new-rate_old) / rate_old < epsilon || iter1 > max_iter
        break;
    end
    rate_old = rate_new;
end

plot(0:iter1-1,rate,'r-o')
grid on
xlabel('Iterations')
ylabel('Sum rate (bits per channel use)')
set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1)
title('MIMO-IFC, K=4, T=3, R=2, \epsilon=1e-3','Interpreter','tex')
%title('SISO-IFC, K=3, T=1, R=1, \epsilon=1e-3','Interpreter','tex')

% 子函数（优化接收矩阵U）
function U = find_U(H,V,sigma2,R,I,K,d)
    J = cell(I,K);
    J(:)={zeros(R,R)};
    U = cell(I,K);
    U(:)={zeros(R,d)};
    for i=1:I
        for k=1:K
            for j=1:K
                for l=1:I
                    J{i,k} = J{i,k} + H{i,k,j}*V{l,j}*(V{l,j}')*(H{i,k,j}'); % 算法Table I, 第四行括号求和的部分
                end
            end
            J{i,k} = J{i,k} + sigma2*eye(R); % 算法Table I, 第四行括号求和加上噪声项的部分
            U{i,k} = J{i,k}\H{i,k,k}*V{i,k}; % 算法Table I, 第四行括号求逆，然后乘以H乘以V
        end
    end     
end



% 子函数（优化权重矩阵W）
function W = find_W(U,H,V,I,K,d)
    W = cell(I,K);
    W(:) = {zeros(d,d)};
    for i=1:I
        for k=1:K
            W{i,k} = inv(eye(d)-U{i,k}'*H{i,k,k}*V{i,k}); % 算法Table I, 第五行
        end
    end
end








%子函数（优化发射波束V）

function V = find_V(alpha1,H,U,W,T,I,K,P)
    V = cell(I,K);
    A = cell(K);
    A(:) = {zeros(T,T)};
    

    for k=1:K   % 公式15括号内求和部分     
        for j=1:K
            for l=1:I
                A{k} = A{k} + alpha1(l,j)*H{l,j,k}'*U{l,j}*W{l,j}*(U{l,j}')*H{l,j,k};
            end
        end   
    end 
    
    max_iter = 100; % 二分法查找最优对偶变量\mu
    mu = zeros(K,1);
    for k=1:K % 对每个基站迭代寻找最优\mu
        mu_min = 0;
        mu_max = 10;
        iter = 0;
        while(1)
            mu1 = (mu_max+mu_min) / 2;
            P_tem = 0;
            for i=1:I % 计算功率和
                V_tem = inv((A{k}+mu1*eye(T)))*alpha1(i,k)*((H{i,k,k}')*U{i,k}*W{i,k}); % 公式15
                P_tem = P_tem + real(trace(V_tem*V_tem'));
            end
            if P_tem > P
                mu_min = mu1;
            else
                mu_max = mu1;
            end
            iter = iter + 1;

            if abs(mu_max - mu_min) < 1e-5 || iter > max_iter
                break
            end
        end
        mu(k) = mu1;
    end

    for i=1:I
        for k=1:K
            V{i,k} =  inv((A{k}+mu(k)*eye(T)))*alpha1(i,k)*((H{i,k,k}')*U{i,k}*W{i,k}); % 公式15
        end 
    end 
end






%子函数（计算系统和速率sum_rate）

function system_rate = sum_rate(H,V,sigma2,R,I,K,alpha1)
    rate = zeros(I,K);
    for i=1:I
        for k = 1:K
            temp = zeros(R,R);
            for l=1:I
                for j=1:K
                    if l~=i || j~=k
                        temp = temp + H{i,k,j}*V{l,j}*(V{l,j}')*(H{i,k,j}');
                    end
                end
            end
            rate(i,k) = log2(det(eye(R)+H{i,k,k}*V{i,k}*(V{i,k}')*(H{i,k,k}')*inv(temp + sigma2*eye(R)))); % 公式2
        end
    end
    system_rate = real(sum(rate.*alpha1,'all'));
end







%
































