%% 严格按 (41)-(43) 实现 CRB(θ) 的通用程序（适用于任意 M）
% 对应图8：M=2（可改为任意值），L=2，θ1=0°，θ2=5,10,15°，SNR=0dB
% 使用分块 FIM (J_θθ, J_θa, J_aa) 和 Schur 补，避免完整 6×6 求逆


clc;
clear all;
close all;
addpath('./functions');


%% 用户参数
M = 10;                          % 阵元数（可修改为任意正整数）
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数
alpha1 = 2; 
alpha2 = 10;         % 复振幅
sigma_w2 = N * abs(alpha1)^2 / SNR_lin;   % 噪声方差

theta1_deg = 0;
theta2_list = 0:0.01:2;
beta_vals = linspace(0, 0.9999, 200);   % 相关系数

% 对称阵列（质心在原点）
n = 0 : (M-1);         % 阵元位置
a = @(th) exp(-1j * 2*pi*d_lambda * n' * sind(th));
da = @(th) -1j * 2*pi*d_lambda * cosd(th) * n' .* a(th);
A = @(th) a(th) * a(th).';
dA = @(th) da(th) * a(th).' + a(th) * da(th).';

% 相干矩阵（通用形式，适用于任意 M）
R_s = @(beta) (1-beta)*eye(M) + beta*ones(M);

factor1 = 2 * N / sigma_w2;       

factor2 = 2 * N * M / sigma_w2;      
% 辅助函数：从复数迹构造 2x2 子块（公式(61)中的块）
blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];

CRB_theta1 = zeros(2, length(theta2_list));

%% 主循环
for k = 1:length(theta2_list)
    theta2_deg = theta2_list(k);

    %% 相干信号
    beta = 1;
    Rs = R_s(beta);
 
    CRB_theta1(1, k) =  CRB(d_lambda, M, theta1_deg, theta2_deg, alpha1, alpha2, Rs, factor1);
    %% 正交信号
    beta = 0;
    Rs = R_s(beta);
    
    CRB_theta1(2, k) =  CRB(d_lambda, M, theta1_deg, theta2_deg, alpha1, alpha2, Rs, factor2);
end

%% 
theta2_list = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];  % 目标2真实角度（度）
MC_trials = 200;                % 蒙特卡洛次数


RMSE_coherent = zeros(size(theta2_list));
RMSE_orth = zeros(size(theta2_list));




%% 绘图
figure(1);
semilogy(theta2_list, CRB_theta1(1,:), 'b-', 'LineWidth',1.5); hold on;
semilogy(theta2_list, CRB_theta1(2,:), 'r--', 'LineWidth',1.5); hold on;
xlabel('\beta'); 
ylabel('CRB on DOA (deg)');
legend('CRB \beta = 1', 'CRB \beta = 0');
grid on;
title(sprintf('Figure 8: M=%d, L=2, SNR=0dB (via (41)-(43))', M));