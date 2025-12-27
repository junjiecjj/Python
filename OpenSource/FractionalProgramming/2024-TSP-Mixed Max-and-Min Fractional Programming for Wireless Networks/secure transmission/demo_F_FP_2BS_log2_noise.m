%% This demo shows how to maximize the sum of secure rates 2 BS with eavesdroppers Fast FP
% This work was done by Yannan CHEN from CUHK (SZ).
% If you have any questions regarding this code, please feel free to contact me at yannanchen@link.cuhk.edu.cn.

clear;clc
N =  2;                % num of Bob and Eve
sigma_E = ones(N);     % noise power at Eve   -80dBm
sigma_B = 1e-1*ones(N);     % noise power at Bob   -90dBm
P_max = 10;                 % power budget for each BS 10dBm
w = ones(N,1);
w(1:N) = 1;
iter_max = 100;
weight = 1;
time = zeros(iter_max,1);
test_num = 1;
test_results = zeros(test_num,1);
H = [1,0.1;0.09,0.87]; % channel of legimate users
h = [0.5,0.11;0.13,0.39];    %  illegimate users
for rand_test = 1:1
    p = ones(N,1)*P_max;
    pre_p = zeros(N,1);
    iter_results = zeros(iter_max+1,1);
    gamma = zeros(N,1); t_gamma = zeros(N,1); % Lagrangian multiplier
    y = zeros(N,1); x = zeros(N,1); % auxiliary variables
    %     tic
    t1 = cputime;
    for iter = 1:iter_max
        [iter_results(iter)] = Compute_obj(sigma_B,sigma_E,p,H,h);
        %% find the optimal gamma
        for n = 1:N
            gamma(n) = H(n,n)*p(n)/((H(n,:)*p-H(n,n)*p(n))+sigma_B(n)); % gamma for Bob
            t_gamma(n) = h(n,n)*p(n)/(h(n,:)*p+sigma_E(n)); % gamma for Eve
        end
        %% find the optimal y and x
        for n = 1:N
            y(n) = sqrt(w(n)*(1+gamma(n))*H(n,n)*p(n))/(H(n,:)*p+sigma_B(n));
            x(n) = sqrt((h(n,:)*p-h(n,n)*p(n))+sigma_E(n))/(w(n)*(1-t_gamma(n))*h(n,n)*p(n));
        end

        cvx_begin
        variable p(N) nonnegative
        expression Bob_R(N)
        expression Eve_IR(N)
        %% 带权重的rate
        for n = 1:N  % 有窃听的SNR
            Bob_R(n) = 2*y(n)*sqrt(w(n)*(1+gamma(n))*H(n,n)*p(n))-y(n)^2*(H(n,:)*p+sigma_B(n));
            Eve_IR(n) = -inv_pos(2*x(n)*sqrt((h(n,:)*p-h(n,n)*p(n))+sigma_E(n))-x(n)^2*w(n)*(1-t_gamma(n))*h(n,n)*p(n));
        end
        maximize sum(Bob_R)+sum(Eve_IR)
        subject to
        p<=ones(N,1)*P_max
        cvx_end
        pre_p = p;
        time(iter) = cputime-t1;
    end
    [iter_results(iter+1)] =  Compute_obj(sigma_B,sigma_E,p,H,h);
    test_results(rand_test) = iter_results(iter+1);
    rand_test
end

