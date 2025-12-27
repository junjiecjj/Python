%% This demo shows how to maximize the sum of secure rates 2 BS with eavesdroppers Direct FP
% This work was done by Yannan CHEN from CUHK (SZ).
% If you have any questions regarding this code, please feel free to contact me at yannanchen@link.cuhk.edu.cn.

clear;clc
N =  2;                     % number of eavesdroppers
sigma_E = ones(N,1);     % noise power at eavesdroppers (Eve)  -80dBm
sigma_B = 1e-1*ones(N,1);     % noise power at users with eavesdroppers (Bob)  -90dBm
P_max = 10;                 % power budget for each BS 10dBm
iter_max = 100;              % max iteration number
H = [1,0.1;0.09,0.87]; % channel of legimate users
h = [0.5,0.11;0.13,0.39];    %  illegimate users
time = zeros(iter_max,1);
p = ones(N,1)*P_max;
pre_p = zeros(N,1);
iter_results = zeros(iter_max+1,1);
t1 = cputime;
[iter_results(1)] = Compute_obj(sigma_B,sigma_E,p,H,h);
for iter = 1:iter_max
    y = zeros(N,1); x = zeros(N,1); 
    %% find optimal auxiliary variables, 'y' is 'y' in the paper, 'x' is 'tilde{y}' in the paper.
    for n = 1:N
        y(n) = sqrt(H(n,n)*p(n))/((H(n,:)*p-H(n,n)*p(n))+sigma_B(n));
        x(n) = sqrt(h(n,:)*p+sigma_E(n))/(h(n,n)*p(n));
    end
    cvx_begin
    variable p(N) nonnegative
    expression Bob_R(N)
    expression Eve_NR(N) % the negative of Eve's data rate
    for n = 1:N
        Bob_R(n) = log(1+2*y(n)*sqrt(H(n,n)*p(n))-y(n)^2*((H(n,:)*p-H(n,n)*p(n))+sigma_B(n)));
        Eve_NR(n) = log(1-inv_pos(2*x(n)*sqrt(h(n,:)*p+sigma_E(n))-x(n)^2*(h(n,n)*p(n))));
    end

    maximize sum(Bob_R)+sum(Eve_NR)
    subject to
    p<=ones(N,1)*P_max
    cvx_end
    [iter_results(iter+1)] = Compute_obj(sigma_B,sigma_E,p,H,h); %% 
    if iter_results(iter+1)<iter_results(iter) %% 
            break
        else
            opt_p = p;
    end
    time(iter) = cputime-t1;
end
