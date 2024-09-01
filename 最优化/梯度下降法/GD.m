

% https://mp.weixin.qq.com/s?__biz=Mzg5MjI1NTAzOA==&mid=2247484716&idx=1&sn=1c57df926f793082f5b258d72a9e6d5e&chksm=ce940e1bc30bb47e4395385a9286a755f88a418fc2e8ea260449179619354393ee539ed91658&mpshare=1&scene=1&srcid=0901pfvvk0BUlQd1xBe5ak6O&sharer_shareinfo=e194e0bff252d167b24e6d9868e64bd3&sharer_shareinfo_first=e194e0bff252d167b24e6d9868e64bd3&exportkey=n_ChQIAhIQ36CZNdzwDzdHdFv%2B%2FUMLvBKfAgIE97dBBAEAAAAAAH6QBbNuK1YAAAAOpnltbLcz9gKNyK89dVj0l8fYentQ4hNty6EGKnxN%2F0L44C3AYyW6E%2BsGaneBbFr%2BrxmYd%2BcXQaklUGdc63IAnLcJq8WN2m57hEJLrWL9%2FfTEKefzAXCLl1aReQRDjxuiOuh5X6c1j4Yap1i4ieREdWfXDJ402mHmhdza6lP%2BlJI2cjFAn9hzSY43aeLqkJmwPfKiVZdcVakIyf4PUlAnrof6d%2FNhq7zMG1E4BjqhDNL3bj71CWi1NZQgMh8UFvT0JyrjNf71cSCua5%2Bu8uZcaAchKPG0p7rwZUsWt%2BsSGoWAeh36lwDyjpOXr7IWyUeIYpuI3xDYieG6RUoLIXvDX2AM1N31t9%2FZ&acctmode=0&pass_ticket=z6XCTFyUsTQLec3VjTrQGZUa1b9%2Fn%2Fk1h5XLiajrvQ7M0jUzMwLCCfao6aGd35nL&wx_header=0#rd




clear
clc
N = 1000 ;
iter_max = 1000 ;
sig_max = 1 ;
% 最大奇异值
sig_min = 1 ;
% 最小奇异值
sig_num = linspace(sig_min,sig_max,N) ;
% 生成奇异值，最大奇异值为100，最小奇异值为1
V = diag(sig_num) ;
% 生成奇异值组成的对角矩阵
A_rand = randn(N,N) ;
% 生成随机矩阵
A_orth = orth(A_rand) ;
% 随机矩阵正交化
A = A_orth*V*A_orth^(-1) ;
% 得到设定条件数的矩阵A
cond(A)
% 试一下矩阵A的条件数是否正确
x = randn(N,1) ;
y = A*x ;

[hat_x,error] = opt_gd(y,A,iter_max) ;
% 使用梯度下降法求解y=Ax

figure(1)
plot(error,'m-o')
xlabel('迭代次数')
ylabel('NMSE')
title(['condition number=',num2str(sig_max/sig_min)])



function [hat_x, error] = opt_gd(y,A,iter_max)
% gradient descent algorithm for solving y=Ax
% Louis Zhang 2020.12.10
% email:zhangyanfeng527@gmail.com
n = size(A,2) ;
x0 = zeros(n,1) ;
g0 = A'*(y-A*x0) ;
% 初始化梯度
% 初始化x_0
for k = 1:iter_max
    g1 = A'*(y-A*x0) ;
    % 计算梯度
    %  alpha = (g0'*g0)/(norm(A*g0)^2) ;
    alpha = 0.1 ;
    % 计算步长
    hat_x = x0 + alpha*g1 ;
    % 更新x_k+1
    error(k) = (norm(y-A*hat_x)/norm(y))^2 ;
    % error(k) = (norm(hat_x-x0)/norm(hat_x))^2 ;
    if error(k)<1e-8
        break;
    else
        x0 = hat_x ;
        g0 = g1 ;
    end
end
end



























