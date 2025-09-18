clc; clear; cvx_clear;
% https://www.cnblogs.com/longtianbin/p/17124657.html
%决策变量维度K
K = 3;
r =[20.8204;
    24.8497;
    22.5085];
p = [ 0.97;
        1;
     0.99];
p_c = 1;
c_1 = [ 0.0451;
        0.0408;
        0.0312];
C_2 = [ -0.0010   0.0493   0.0202;
        0.0313   -0.0010    0.0202;
        0.0313    0.0493   -0.0010];   
r_d = [ 5.7464
        8.2832
        9.4225];
c_3 = 8;

% Test.m 脚本 
delta = 1E-6;
IterMax = 20;

%%
c_3_vec = 7.6:0.2:9.2;
L = length(c_3_vec);
result_CC = zeros(L,1);
result_Dinkelbach = zeros(L,1);
result_Quadratic = zeros(L,1);

for j = 1 : L
    [isFeasible,obj_opt,tau_opt_CC] = LFP_Charnes_Cooper(K,r,p,p_c,c_1,C_2,r_d,c_3_vec(j));
    result_CC(j) = obj_opt;
    
    [isFeasible,obj_opt,tau_opt_Dink,Q_log,F_log] = LFP_Dinkelbach(K,r,p,p_c,c_1,C_2,r_d,c_3_vec(j),delta,IterMax);
    result_Dinkelbach(j) = obj_opt;

    [isFeasible,obj_opt,tau_opt_Quad,F_log] = LFP_Quadratic(K,r,p,p_c,c_1,C_2,r_d,c_3_vec(j),delta,IterMax);
    result_Quadratic(j) = obj_opt;
    
    fprintf('# %d \n %s \n %s \n %s \n',j, num2str(tau_opt_CC'), num2str(tau_opt_Dink'), num2str(tau_opt_Quad'))
end
%%
plot(c_3_vec,result_CC,'-o',c_3_vec,result_Dinkelbach,'-p',c_3_vec,result_Quadratic,'-+','MarkerSize',8,'LineWidth',2)
legend({'Charnes-Cooper','Dinkelbach','Quadratic'})
ylabel('最优目标值')
xlabel('c_3')
title('不同变换方法求解的最优目标函数值与参数c_3的关系')

function [isFeasible,obj_opt,tau_opt] = LFP_Charnes_Cooper(K,r,p,p_c,c_1,C_2,r_d,c_3)
%LFP_CHARNES_COOPER 使用Charnes-Cooper变换求解线性分式规划问题
%  输入参数说明：
%       K,r,p,p_c,c_1,C_2,r_d,c_3 优化问题参数
%  输出参数说明：
%       isFeasible 是否可行
%       obj_opt 最优目标函数值
%       tau_opt 最优解
%% 调用CVX求解线性规划
cvx_begin quiet
    variable q(K,1) nonnegative
    variable z nonnegative
    
    maximize (r'*q)
    
    subject to
        q >= z * c_1;
        C_2 * q >= 0;
        r_d'* q >= z * c_3;
        sum(q) <= z;
        p'* q + z * p_c == 1;
cvx_end

%% 结果返回
obj_opt = cvx_optval;
isFeasible = true;
tau_opt = q/z;
if strcmp(cvx_status,'Infeasible') || strcmp(cvx_status,'Unbounded') || isnan(cvx_optval) || isinf(cvx_optval)
    isFeasible = false;
end

end





function [isFeasible,obj_opt,tau_opt,Q_log,F_log] = LFP_Dinkelbach(K,r,p,p_c,c_1,C_2,r_d,c_3,delta,IterMax)
%LFP_DINKELBACH 使用Dinkelbach变换求解线性分式规划问题
%  输入参数说明：
%       K,r,p,p_c,c_1,C_2,r_d,c_3 优化问题参数
%       delta 一个小正数 收敛门限
%       IterMax 最大迭代次数
%  输出参数说明：
%       isFeasible 是否可行
%       obj_opt 最优目标函数值
%       tau_opt 最优解
%       Q_log 记录每次迭代的Q值
%       F_log 记录每次迭代的F值
%% 初始化
Q_log = []; 
F_log = [];
isFeasible = true;
Q = 0;

%% 迭代求解
for j = 1 : IterMax
    cvx_clear
    % 调用CVX
    cvx_begin quiet
        variable tau(K,1) nonnegative
    
        maximize ( r'*tau - Q *(p'*tau + p_c) )
    
        subject to
            tau >=  c_1;
            C_2 * tau >= 0;
            r_d'* tau >=  c_3;
            sum(tau) <= 1;
   cvx_end
   
   % 不可行或者无界则直接退出循环
   if strcmp(cvx_status,'Infeasible') || strcmp(cvx_status,'Unbounded') || isnan(cvx_optval) || isinf(cvx_optval)
        isFeasible = false;
        break
   end
   
   Q_log = [Q_log;Q];
   F_log = [F_log;cvx_optval];

   % 如果收敛则退出循环，如果不收敛则更新Q
   if cvx_optval <= delta
       break
   else
       Q = r'*tau / (p'*tau + p_c);
   end
   
end

%% 返回结果
tau_opt = tau;
obj_opt = r'*tau_opt / (p'*tau_opt + p_c);

end


function [isFeasible,obj_opt,tau_opt,F_log] = LFP_Quadratic(K,r,p,p_c,c_1,C_2,r_d,c_3,delta,IterMax)
%LFP_QUADRATIC 使用Quadratic变换求解线性分式规划问题
%  输入参数说明：
%       K,r,p,p_c,c_1,C_2,r_d,c_3 优化问题参数
%       delta 一个小正数 收敛门限
%       IterMax 最大迭代次数
%  输出参数说明：
%       isFeasible 是否可行
%       obj_opt 最优目标函数值
%       tau_opt 最优解
%       F_log 记录每次迭代的F值
%% 计算初始可行解
%   使用CVX求初始解 一个线性规划可行性问题
    cvx_begin quiet
        variable tau(K,1) nonnegative    
        subject to
            tau >=  c_1;
            C_2 * tau >= 0;
            r_d'* tau >=  c_3;
            sum(tau) <= 1;
   cvx_end
   
%   初始化数据
tau_i = tau;
F_i = r'*tau_i / (p'*tau_i + p_c);
F_log = [F_i];
isFeasible = true;   

%% 迭代求解
%% 迭代求解
for j = 1 : IterMax
    
    y = sqrt(r'*tau_i)/(p'*tau_i + p_c);
    cvx_clear
    % 调用CVX
    cvx_begin quiet
        variable tau(K,1) nonnegative
    
        maximize ( 2*y*sqrt(r'*tau) -  y^2*(p'*tau + p_c) )
    
        subject to
            tau >=  c_1;
            C_2 * tau >= 0;
            r_d'* tau >=  c_3;
            sum(tau) <= 1;
   cvx_end
   tau_i_1 = tau;
   F_i_1 = r'*tau_i_1 / (p'*tau_i_1 + p_c);
   F_log = [F_log;F_i_1];
   
   % 不可行或者无界则直接退出循环
   if strcmp(cvx_status,'Infeasible') || strcmp(cvx_status,'Unbounded') || isnan(cvx_optval) || isinf(cvx_optval)
        isFeasible = false;
        break
   end
   
   % 如果收敛则退出循环，如果不收敛则更新Q
   if (F_i_1 - F_i) <= delta
       break
   else
       tau_i = tau_i_1;
       F_i = F_i_1;
   end
   
end

obj_opt = F_i_1;
tau_opt = tau_i_1;

end

