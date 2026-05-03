%% Figure 9 完整复现：CRB + ML 估计 RMSE（基于原文(27)-(28)）
% M=10, L=2, θ1=0°, θ2 varies, SNR=0dB, N=1

clear; clc; close all;

%% 系统参数
M = 10;                         % 阵元数
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 基础 SNR (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数（为简化，CRB与N成反比可归一化）
alpha_true = [1; 1];            % 两个目标的真实复振幅（设为1）
sigma2_coherent = 1/SNR_lin;    % 相干信号噪声方差
sigma2_orth = sigma2_coherent / M;   % 正交信号 TOT 补偿

theta1_true = 0;                % 目标1真实角度（度）
theta2_list = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0];  % 目标2真实角度（度）
MC_trials = 200;                % 蒙特卡洛次数

% 对称阵列（质心原点）
n = 0 : (M-1);
a = @(th) exp(-1j * pi *  n' * sind(th));
da = @(th) -1j * pi * cosd(th) * n' .* a(th);
A = @(th) a(th) * a(th).';
dA = @(th) da(th) * a(th).' + a(th) * da(th).';

% 预分配结果
CRB_coherent = zeros(size(theta2_list));
CRB_orth = zeros(size(theta2_list));
RMSE_coherent = zeros(size(theta2_list));
RMSE_orth = zeros(size(theta2_list));

%% 辅助函数：计算 CRB（使用等效导向矢量法，6x6 FIM 求逆）
function CRB_deg = crb_two_targets(theta1, theta2, alpha, sigma2, M, N, R_s)
    n = 0:M-1;
    a_func = @(th) exp(-1j * pi * n' * sind(th));
    da_func = @(th) -1j * pi * cosd(th) * n' .* a_func(th);
    A = @(th) a_func(th) * a_func(th).';
    dA = @(th) da_func(th) * a_func(th).' + a_func(th) * da_func(th).';
    [U, Lambda] = eig(R_s);
    lambda = diag(Lambda);
    U_sqrtL = U * diag(sqrt(max(lambda,0)));
    d_beta = @(th) reshape( sqrt(N) * (A(th) * U_sqrtL), [], 1);
    d_beta_deriv = @(th) reshape( sqrt(N) * (dA(th) * U_sqrtL), [], 1);
    d1 = d_beta(theta1); 
    d2 = d_beta(theta2);
    d1p = d_beta_deriv(theta1); 
    d2p = d_beta_deriv(theta2);
    G = [alpha(1)*d1p, alpha(2)*d2p, d1, 1j*d1, d2, 1j*d2];
    J = (2/sigma2) * real(G'*G);
    CRB_rad2 = inv(J(1,1));
    CRB_deg = sqrt(CRB_rad2) * 180/pi;
end

%% 计算 CRB
for k = 1:length(theta2_list)
    theta2 = theta2_list(k);
    % 相干信号
    R_s_coherent = a(theta1_true) * a(theta1_true)';
    CRB_coherent(k) = crb_two_targets(theta1_true, theta2, alpha_true, sigma2_coherent, M, N, R_s_coherent);
    % 正交信号
    R_s_orth = eye(M);
    CRB_orth(k) = crb_two_targets(theta1_true, theta2, alpha_true, sigma2_orth, M, N, R_s_orth);
end

%% ML 蒙特卡洛仿真
for k = 1:length(theta2_list)
    theta2_true = theta2_list(k);
    fprintf('Processing θ2 = %.1f° ...\n', theta2_true);
    
    % 发射信号（相干信号固定，正交信号随机，但需要与 eta 计算一致）
    R_s_coherent = a(theta1_true) * a(theta1_true)';
    % 对于相干信号，发射信号 s = a(theta1_true) （所有快拍相同，N=1）
    s_coherent = a(theta1_true);
    
    theta1_est_coherent = zeros(MC_trials, 1);
    theta1_est_orth = zeros(MC_trials, 1);
    
    for mc = 1:MC_trials
        % ----- 相干信号 -----
        % 生成接收数据
        A1 = A(theta1_true);
        A2 = A(theta2_true);
        s = s_coherent;
        signal = alpha_true(1) * A1 * s + alpha_true(2) * A2 * s;
        w = sqrt(sigma2_coherent/2) * (randn(M,1) + 1j*randn(M,1));
        y = signal + w;
        % 计算 eta
        eta = compute_eta(y, s, R_s_coherent, N);
        % 二维网格搜索最大化 L(θ) = eta^H P_D eta
        % 搜索范围：θ1 ∈ [-0.8,0.8], θ2 ∈ [θ2_true-0.8, θ2_true+0.8]，步长0.02
        theta1_grid = linspace(-0.8, 0.8, 81);
        theta2_grid = linspace(theta2_true-0.8, theta2_true+0.8, 81);
        best_val = -inf;
        best_theta1 = NaN;
        for i = 1:length(theta1_grid)
            th1 = theta1_grid(i);
            for j = 1:length(theta2_grid)
                th2 = theta2_grid(j);
                D = construct_D([th1, th2], R_s_coherent, M, N, a);
                P_D = D * ((D'*D) \ D');
                L_val = real(eta' * P_D * eta);
                if L_val > best_val
                    best_val = L_val;
                    best_theta1 = th1;
                end
            end
        end
        theta1_est_coherent(mc) = best_theta1;
        
        % ----- 正交信号 -----
        % 每次独立生成发射信号 s_orth，满足 E[s s^H] = I
        s_orth = sqrt(M) * (randn(M,1) + 1j*randn(M,1)) / sqrt(2);
        signal_orth = alpha_true(1) * A1 * s_orth + alpha_true(2) * A2 * s_orth;
        w_orth = sqrt(sigma2_orth/2) * (randn(M,1) + 1j*randn(M,1));
        y_orth = signal_orth + w_orth;
        R_s_orth = eye(M);
        eta_orth = compute_eta(y_orth, s_orth, R_s_orth, N);
        % 网格搜索
        best_val = -inf;
        best_theta1_orth = NaN;
        for i = 1:length(theta1_grid)
            th1 = theta1_grid(i);
            for j = 1:length(theta2_grid)
                th2 = theta2_grid(j);
                D = construct_D([th1, th2], R_s_orth, M, N, a);
                P_D = D * ((D'*D) \ D');
                L_val = real(eta_orth' * P_D * eta_orth);
                if L_val > best_val
                    best_val = L_val;
                    best_theta1_orth = th1;
                end
            end
        end
        theta1_est_orth(mc) = best_theta1_orth;
    end
    
    RMSE_coherent(k) = sqrt(mean((theta1_est_coherent - theta1_true).^2));
    RMSE_orth(k) = sqrt(mean((theta1_est_orth - theta1_true).^2));
end

%% 绘图
figure(1);
semilogy(theta2_list, CRB_coherent, 'b-', 'LineWidth', 2); hold on;
semilogy(theta2_list, CRB_orth, 'r--', 'LineWidth', 2); hold on;
semilogy(theta2_list, RMSE_coherent, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b'); hold on;
semilogy(theta2_list, RMSE_orth, 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Separation angle \theta_2 (deg)');
ylabel('RMSE / CRB (deg)');
legend('Coherent CRB', 'Orthogonal CRB', 'Coherent ML', 'Orthogonal ML');
grid on;
title('Figure 9: M=10, L=2, SNR=0dB');

%% 辅助函数：计算等效观测 eta（基于充分统计量）
% 输入：y (Mx1), s (Mx1), R_s, N=1
% 输出：eta (M^2 x 1)
function eta = compute_eta(y, s, R_s, N)
    E = (1/sqrt(N)) * (y * s');   % MxM
    [U, Lambda] = eig(R_s);
    lambda = diag(Lambda);
    Lambda_inv_sqrt = diag(1 ./ sqrt(max(lambda, eps)));
    U_tmp = U * Lambda_inv_sqrt;
    eta = reshape(E * U_tmp, [], 1);
end

%% 辅助函数：构造 D 矩阵 (M^2 x L)
function D = construct_D(theta_vec, R_s, M, N, a)
    n = 0 : (M-1);
    a_func = @(th) exp(-1j * pi * n' * sind(th));
    A = @(th) a_func(th) * a_func(th).';
    [U, Lambda] = eig(R_s);
    lambda = diag(Lambda);
    U_sqrtL = U * diag(sqrt(max(lambda,0)));
    L = length(theta_vec);
    D = zeros(M^2, L);
    for l = 1:L
        D(:,l) = reshape( sqrt(N) * (A(theta_vec(l)) * U_sqrtL), [], 1);
    end
end

