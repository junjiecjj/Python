


%% 单目标下的CRB和ML对比，正交信号下死活对不上，相干信号下可以对得上


clc;
clear all;
close all;
addpath('./functions');

%% 用户参数
M = 11;                         % 阵元数（可修改为任意正整数）
d_lambda = 0.5;                 % 半波长间距
SNR_dB = -4:4:16;                % 信噪比 (dB)
N = 1;                          % 快拍数
alpha = 1;
theta1 = 0;  % 单位:度

% 蒙特卡洛参数
MC_trials = 100;            % 每个 SNR 点的仿真次数（可增加）
init_range = 0.5;           % 初始搜索半径（度）
final_tol = 1e-4;           % 最终精度（度）

% 对称阵列（质心在原点）
n = -(M-1)/2 : (M-1)/2;         % 阵元位置
a = @(th) exp(-1j * 2 * pi * d_lambda * n' * sind(th));
da = @(th) -1j * 2 * pi * d_lambda * cosd(th) * n' .* a(th);
A = @(th) a(th) * a(th).';
dA = @(th) da(th) * a(th).' + a(th) * da(th).';

% 相干信号发射导向矢量（波束指向目标）
s_coherent = a(theta1);
R_s_coherent = a(theta1) * a(theta1)';

% 正交信号发射矢量生成函数 (单快拍，满足 E[s s^H]=I)
% s_orth_gen = @() sqrt(M) * (randn(M,1) + 1j*randn(M,1)) / sqrt(2);
s_orth = (randn(M,1) + 1j*randn(M,1)) / sqrt(2*N);
R_s_orth = eye(M);

CRB_coherent = zeros(1, length(SNR_dB));
CRB_orth = zeros(1, length(SNR_dB));
RMSE_coherent = zeros(size(SNR_dB));
RMSE_orth = zeros(size(SNR_dB));


%% 主循环
for idx = 1:length(SNR_dB)
    SNR_db = SNR_dB(idx);
    SNR_lin = 10^(SNR_db/10);

    %% 相干信号: R_s = a(θ) a^H(θ)：发射增益已隐含在R_s中，故SNR使用基础值
    SNR_coherent = SNR_lin;
    sigma_w2_coherent = N * abs(alpha)^2 ./ SNR_coherent;   % 噪声方差

    a_vec = a(theta1);
    a_dot = da(theta1);

    R_s = a_vec * a_vec';          % M×M
    R_sT = R_s.';                  % 转置（不共轭）
    
    % 计算公式(44)中的各项
    term1 = M * (a_dot' * R_sT * a_dot);
    term2 = (a_vec' * R_sT * a_vec) * (a_dot' * a_dot);
    % term3 = M * abs(a_vec' * R_sT * a_dot)^2 / (a_vec' * R_sT * a_vec);
    term3 =  M * abs(a_vec.' * R_s * conj(a_dot))^2 / (a_vec' * R_sT * a_vec);

    Denom = real(term1 + term2 - term3);   % 应为正实数
    CRB_var_rad2 = 1 / (2 * SNR_coherent * Denom);
    CRB_coherent(idx) = sqrt(CRB_var_rad2) * (180/pi);

    %% 正交信号由于TOT补偿，有效SNR提高M倍
    SNR_orth = M * SNR_lin;
    sigma_w2_orth = N * abs(alpha)^2 ./ SNR_orth;   % 正交信号 TOT 补偿

    a_vec = a(theta1);
    a_dot = da(theta1);
    
    R_s = eye(M);
    R_sT = eye(M);               % 单位阵转置仍是单位阵
    
    term1 = M * (a_dot' * R_sT * a_dot);
    term2 = (a_vec' * R_sT * a_vec) * (a_dot' * a_dot);
    % term3 = M * abs(a_vec' * R_sT * a_dot)^2 / (a_vec' * R_sT * a_vec);
    term3 =  M * abs(a_vec.' * R_s * conj(a_dot))^2 / (a_vec' * R_sT * a_vec);
    Denom = real(term1 + term2 - term3);
    % 注意此处使用补偿后的 SNR_orth
    CRB_var_rad2 = 1 / (2 * SNR_orth * Denom);
    CRB_orth(idx) = sqrt(CRB_var_rad2) * (180/pi);

    % ----- 蒙特卡洛 ML 估计 -----
    fprintf('SNR = %.1f dB: ', SNR_db);
    
    % 相干信号
    theta_est_c = zeros(MC_trials, 1);
    for mc = 1:MC_trials
        % 生成接收数据 y = α A s + w
        signal = alpha * A(theta1) * s_coherent;
        w = sqrt(sigma_w2_coherent/2) * (randn(M,1) + 1j*randn(M,1));
        y = signal + w;
        eta = compute_eta(y, s_coherent, R_s_coherent, N);
        theta_est_c(mc) = ml_single(eta, R_s_coherent, M, N, 0.5, 1e-4);
    end
    RMSE_coherent(idx) = sqrt(mean((theta_est_c - theta1).^2));
    
    fprintf('CRB_coherent =%.4f, coherent RMSE=%.4f rad, ', CRB_coherent(idx), RMSE_coherent(idx));
    
    % 正交信号
    theta_est_o = zeros(MC_trials, 1);
    for mc = 1:MC_trials
        s_orth = (randn(M,1) + 1j*randn(M,1)) / sqrt(2);
        signal = alpha * A(theta1) * s_orth;
        w = sqrt(sigma_w2_orth/2) * (randn(M,1) + 1j*randn(M,1));
        y = signal + w;
        eta = compute_eta(y, s_orth, R_s_orth, N);
        theta_est_o(mc) = ml_single(eta, R_s_orth, M, N, 0.5, 1e-4);
    end
    RMSE_orth(idx) = sqrt(mean((theta_est_o - theta1).^2));
    
    fprintf('CRB_orth = %.4f, orthogonal RMSE=%.4f rad\n', CRB_orth(idx), RMSE_orth(idx));

end

%% 绘图
figure(1);
semilogy(SNR_dB, CRB_coherent, 'b-', 'LineWidth', 1.5); hold on;
semilogy(SNR_dB, RMSE_coherent, 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
semilogy(SNR_dB, CRB_orth, 'r--', 'LineWidth', 1.5);
semilogy(SNR_dB, RMSE_orth, 'r^', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
xlabel('SNR (dB)');
ylabel('RMSE / CRB (rad)');
legend('Coherent CRB', 'Coherent ML', 'Orthogonal CRB', 'Orthogonal ML');
title(sprintf('Single Target ML vs CRB (M=%d, θ=0°)', M));
grid on; grid minor;


%% 辅助函数：计算 eta (公式(10)及后续白化)
function eta = compute_eta(y, s, R_s, N)
    E = (1/sqrt(N)) * (y * s');
    [U, Lambda] = eig(R_s);
    lambda = diag(Lambda);
    tol = 1e-12;
    lambda_inv_sqrt = zeros(size(lambda));
    valid = lambda > tol;
    lambda_inv_sqrt(valid) = 1 ./ sqrt(lambda(valid));
    U_tmp = U * diag(lambda_inv_sqrt);
    eta = reshape(E * U_tmp, [], 1);
end

%% 辅助函数：等效导向矢量 d_beta (公式(16))
function d = d_beta(th, R_s, M, N)
    n = -(M-1)/2 : (M-1)/2;
    a = @(th) exp(-1j * pi * n' * sind(th));
    A = a(th) * a(th).';
    [U, Lambda] = eig(R_s);
    lambda = diag(Lambda);
    lambda_sqrt = sqrt(max(lambda, 0));
    U_sqrtL = U * diag(lambda_sqrt);
    d = reshape( sqrt(N) * (A * U_sqrtL), [], 1);
end

%% 辅助函数：一维迭代细化 ML 估计（基于 (29)-(33) 单目标形式）
function theta_est = ml_single(eta, R_s, M, N, init_range, final_tol)
    if nargin < 5, init_range = 0.5; end
    if nargin < 6, final_tol = 1e-4; end
    center = 0;   % 真实角度在 0° 附近
    range = init_range;
    while range > final_tol
        grid = linspace(center - range, center + range, 101);
        L_vals = zeros(size(grid));
        for i = 1:length(grid)
            d = d_beta(grid(i), R_s, M, N);
            L_vals(i) = abs(d' * eta)^2 / (d' * d);
        end
        [~, idx] = max(L_vals);
        center = grid(idx);
        range = range / 2;
    end
    theta_est = center;
end
