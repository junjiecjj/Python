%% 复现 Bekkerman & Tabrikian (2006) Figure 10 - 修正版
% 检测性能 ROC: M=10, 单目标 θ=20°, SNR=10dB
% 相干信号: 发射波束指向 0° (波束形状损失)
% 正交信号: 全向发射，TOT 补偿 (有效 SNR 提高 M 倍)


clc;
clear all;
close all;
% addpath('./functions');

rng(42); 


%% 参数
M = 10;                     % 阵元数
d_lambda = 0.5;             % 半波长间距
theta_t = 20;               % 目标方向 (度)
theta_beam = 0;             % 相干发射波束指向 (度)
SNR_dB = 10;                % 基础信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 10;                     % 快拍数
alpha = 1;                  % 目标复幅度

% 噪声方差
sigma_w2_coherent = 1 / SNR_lin;          % 相干信号
sigma_w2_orth = sigma_w2_coherent / M;    % 正交信号 (TOT补偿)

% 对称阵列 (质心原点)
idx = -(M-1)/2 : (M-1)/2;          % -4.5:4.5
a = @(theta) exp(-1j * pi * d_lambda * idx' * sind(theta));

% 相干发射导向矢量
a_beam = a(theta_beam);
R_s_coherent = a_beam * a_beam';
R_sT_coherent = R_s_coherent.';

% 正交发射信号矩阵 (M x N)，列正交且每列功率 M
% 使用 DFT 矩阵生成正交波形 (复指数)
S_orth = sqrt(M) * dftmtx(M) / sqrt(M);   % 归一化 DFT 矩阵，每列范数 sqrt(M)
S_orth = S_orth(:, 1:N);                  % 取前 N 列
% 验证: S_orth * S_orth' = N * eye(M) (当 N=M)
if N == M
    fprintf('正交发射矩阵验证: S S^H = %.2f * I\n', max(abs(S_orth*S_orth' - N*eye(M)), [], 'all'));
end

% 相干发射信号矩阵 (所有快拍相同)
S_coherent = a_beam * ones(1, N);

%% 蒙特卡洛参数
numMC = 2000;               % 各假设下仿真次数
thresh_num = 200;           % 阈值点数
thresh_list = linspace(0, 30, thresh_num);

% 存储统计量
Lc0 = zeros(1, numMC);   % 相干 H0
Lc1 = zeros(1, numMC);   % 相干 H1
Lo0 = zeros(1, numMC);   % 正交 H0
Lo1 = zeros(1, numMC);   % 正交 H1

% 辅助函数：单目标 GLRT 统计量
% 输入: Y (MxN), S (MxN), R_sT (MxM)
% 输出: max_θ [ |a^H(θ) E a^*(θ)|^2 / (M a^H(θ) R_sT a(θ)) ]
function L = glrt_stat(Y, S, R_sT, a_func)
    N = size(Y,2);
    E = (1/sqrt(N)) * Y * S';        % 充分统计量矩阵 MxM
    % 搜索角度网格
    theta_grid = -60:0.2:60;         % 度，步长0.2°精度足够
    L_vals = zeros(size(theta_grid));
    for k = 1:length(theta_grid)
        th = theta_grid(k);
        a_th = a_func(th);
        num = abs(a_th' * E * conj(a_th))^2;
        den = size(a_th,1) * (a_th' * R_sT * a_th);
        L_vals(k) = num / den;
    end
    L = max(L_vals);
end

%% 蒙特卡洛仿真
fprintf('蒙特卡洛仿真 (%d 次)...\n', numMC);
parfor mc = 1:numMC      % 可使用 parfor 加速
    % 噪声
    w_coherent = sqrt(sigma_w2_coherent/2) * (randn(M,N) + 1j*randn(M,N));
    w_orth = sqrt(sigma_w2_orth/2) * (randn(M,N) + 1j*randn(M,N));
    
    % 目标响应矩阵 A = a(θ_t) * a(θ_t)^T
    A_target = a(theta_t) * a(theta_t).';
    
    % 相干 H0
    Y_c0 = w_coherent;
    Lc0(mc) = glrt_stat(Y_c0, S_coherent, R_sT_coherent, a);
    % 相干 H1
    Y_c1 = alpha * A_target * S_coherent + w_coherent;
    Lc1(mc) = glrt_stat(Y_c1, S_coherent, R_sT_coherent, a);
    
    % 正交 H0
    Y_o0 = w_orth;
    Lo0(mc) = glrt_stat(Y_o0, S_orth, eye(M).', a);   % R_sT = I
    % 正交 H1
    Y_o1 = alpha * A_target * S_orth + w_orth;
    Lo1(mc) = glrt_stat(Y_o1, S_orth, eye(M).', a);
end

%% 计算经验 ROC
Pd_c = zeros(size(thresh_list));
Pf_c = zeros(size(thresh_list));
Pd_o = zeros(size(thresh_list));
Pf_o = zeros(size(thresh_list));

for i = 1:length(thresh_list)
    gamma = thresh_list(i);
    Pf_c(i) = mean(Lc0 >= gamma);
    Pd_c(i) = mean(Lc1 >= gamma);
    Pf_o(i) = mean(Lo0 >= gamma);
    Pd_o(i) = mean(Lo1 >= gamma);
end

%% 正交信号理论渐近 ROC (非中心卡方分布, 自由度2)
% 从 Lo1 统计量中估计非中心参数 λ = mean(Lo1) - 2 (因为中心卡方均值为2)
lambda_est = max(0, mean(Lo1) - 2);
fprintf('正交信号 H1 统计量均值 = %.2f, 估计 λ = %.2f\n', mean(Lo1), lambda_est);
thresh_theory = linspace(0, 30, 500);
Pfa_theory = 1 - chi2cdf(thresh_theory, 2);
Pd_theory = 1 - ncx2cdf(thresh_theory, 2, lambda_est);

%% 绘图
figure('Position', [100, 100, 600, 500]);
plot(Pf_c, Pd_c, 'b-', 'LineWidth', 2, 'DisplayName', 'Coherent (sim.)');
hold on;
plot(Pf_o, Pd_o, 'r--', 'LineWidth', 2, 'DisplayName', 'Orthogonal (sim.)');
plot(Pfa_theory, Pd_theory, 'r:', 'LineWidth', 2, 'DisplayName', 'Orthogonal (asymp.)');
xlabel('Probability of False Alarm P_{fa}');
ylabel('Probability of Detection P_d');
title('Figure 10: M=10, L=1, θ=20°, SNR=10dB');
grid on;
set(gca, 'XScale', 'log');
xlim([1e-3, 1]);
ylim([0, 1]);
legend('Location', 'southeast');
hold off;

%% 打印统计量均值
fprintf('相干: H0 均值=%.3f, H1 均值=%.3f\n', mean(Lc0), mean(Lc1));
fprintf('正交: H0 均值=%.3f, H1 均值=%.3f\n', mean(Lo0), mean(Lo1));