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
sigma2_coherent = N * abs(alpha1)^2 / SNR_lin;   % 噪声方差
sigma2_orth = sigma2_coherent;   % 正交信号 TOT 补偿

theta1_true = 0;
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

factor1 = 2 * N / sigma2_coherent;
factor2 = 2 * N *M / sigma2_coherent;

% 辅助函数：从复数迹构造 2x2 子块（公式(61)中的块）
blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];
CRB_theta1 = zeros(2, length(theta2_list));

%% 主循环
for k = 1:length(theta2_list)
    theta2_deg = theta2_list(k);

    %% 相干信号
    beta = 1;
    Rs = R_s(beta);
    CRB_theta1(1, k) =  CRB(d_lambda, M, theta1_true, theta2_deg, alpha1, alpha2, Rs, factor1);
    %% 正交信号
    beta = 0;
    Rs = R_s(beta);
    CRB_theta1(2, k) =  CRB(d_lambda, M, theta1_true, theta2_deg, alpha1, alpha2, Rs, factor2);
end

%% 
theta2_list_scat = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];  % 目标2真实角度（度）
RMSE_coherent = zeros(size(theta2_list_scat));
RMSE_orth = zeros(size(theta2_list_scat));
MC_trials = 100;                % 蒙特卡洛次数

for k = 1:length(theta2_list_scat)
    theta2_true = theta2_list_scat(k);
    % fprintf('Processing θ2 = %.1f° ...\n', theta2_true);
    
    % 发射信号（相干信号固定，正交信号随机，但需要与 eta 计算一致）
    R_s_coherent = a(theta1_true) * a(theta1_true)';
    % 对于相干信号，发射信号 s = a(theta1_true) （所有快拍相同，N=1）
    s_coherent = a(theta1_true);
    
    theta1_est_coherent = zeros(MC_trials, 1);
    theta1_est_orth = zeros(MC_trials, 1);
    
    for mc = 1:MC_trials
        fprintf('  Processing: θ2 = %.1f°, %d', theta2_true, mc);
        fprintf('\r');
        A1 = A(theta1_true);
        A2 = A(theta2_true);
        % ----- 相干信号 -----
        % 生成接收数据
        s = s_coherent;
        alphaAs = (alpha1 * A1 + alpha2 * A2) * s;
        w = sqrt(sigma2_coherent/2) * (randn(M,1) + 1j*randn(M,1));
        y = alphaAs + w;
        % 计算 eta
        eta_coherent = compute_eta(y, s, R_s_coherent, N);
        % 二维网格搜索最大化 L(θ) = eta^H P_D eta
        theta1_est_coherent(mc) = search_theta1_iterative(eta_coherent, R_s_coherent, M, N, theta2_true);

        % ----- 正交信号 -----
        % 每次独立生成发射信号 s_orth，满足 E[s s^H] = I
        s_orth = sqrt(M) * (randn(M,1) + 1j*randn(M,1)) / sqrt(2);
        alphaAs_orth = (alpha1 * A1 + alpha2 * A2) * s_orth;
        w_orth = sqrt(sigma2_orth/2) * (randn(M,1) + 1j*randn(M,1));
        y_orth = alphaAs_orth + w_orth;
        R_s_orth = eye(M);
        eta_orth = compute_eta(y_orth, s_orth, R_s_orth, N);
        % % 网格搜索
        theta1_est_orth(mc) = search_theta1_iterative(eta_orth, R_s_orth, M, N, theta2_true);
    end
    
    RMSE_coherent(k) = sqrt(mean((theta1_est_coherent - theta1_true).^2));
    RMSE_orth(k) = sqrt(mean((theta1_est_orth - theta1_true).^2));
end

%% 绘图
figure(1);
semilogy(theta2_list, CRB_theta1(1,:), 'b-', 'LineWidth',1.5); hold on;
semilogy(theta2_list, CRB_theta1(2,:), 'r--', 'LineWidth',1.5); hold on;
semilogy(theta2_list_scat, RMSE_coherent, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b'); hold on;
semilogy(theta2_list_scat, RMSE_orth, 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('\beta'); 
ylabel('CRB on DOA (deg)');
legend('CRB \beta = 1', 'CRB \beta = 0', 'Coherent ML', 'Orthogonal ML');
grid on;
title(sprintf('Figure 8: M=%d, L=2, SNR=0dB (via (41)-(43))', M));

function theta1_est = search_theta1_iterative(eta, R_s, M, N, theta2_true, init_range, final_tol, npoints)
% 迭代细化一维搜索，估计 theta1（已知 theta2）
% 参照 MUSIC 算法的迭代峰值搜索思路
% 输入：
%   eta          - 等效观测向量 (M^2 x 1)
%   R_s          - 发射相干矩阵
%   M,N          - 阵元数、快拍数
%   theta2_true  - 已知的第二个目标角度（度）
%   init_range   - 初始搜索半径（度），默认 0.5
%   final_tol    - 最终搜索半径阈值（度），默认 0.0001
%   npoints      - 每次迭代的网格点数，默认 21（奇数，包含中心点）
% 输出：
%   theta1_est   - 估计的第一个目标角度（度）
    final_tol = 1e-4;
    npoints = 21;
    center = 0;      % 初始中心
    range = 0.1;
    
    while range > final_tol
        % 生成当前搜索区域的均匀网格
        theta1_vec = linspace(center - range, center + range, npoints);
        L_vals = zeros(size(theta1_vec));
        
        for i = 1:length(theta1_vec)
            th1 = theta1_vec(i);
            D = construct_D([th1, theta2_true], R_s, M, N);
            P_D = D * ((D'*D) \ D');
            L_vals(i) = real(eta' * P_D * eta);
        end
        
        % 找到最大似然对应的角度
        [~, idx] = max(L_vals);
        center = theta1_vec(idx);
        
        % 缩小搜索半径
        range = range / 2;
    end
    theta1_est = center;
end