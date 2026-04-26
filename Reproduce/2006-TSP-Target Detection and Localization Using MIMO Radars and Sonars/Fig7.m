
clc;
clear all;
close all;
% addpath('./functions');

rng(42); 

%% 复现 Bekkerman & Tabrikian (2006) Figure 7 - 严格匹配版
% M=10, ULA, 半波长间距, 阵列对称(质心为原点)
% 相干信号: R_s = a(θ)a^H(θ), SNR = 基础值 (0dB)
% 正交信号: R_s = I, 因 TOT 补偿 SNR = M * 基础值, 且采用小角度近似使 CRB 水平

%% 参数
M = 10;                     % 阵元数
d_lambda = 0.5;             % 半波长间距
SNR_dB = 0;                 % 基础信噪比 (0 dB)
SNR_lin = 10^(SNR_dB/10);   % 线性值

% 正交信号有效 SNR (TOT 补偿)
SNR_orth = M * SNR_lin;
% 相干信号 SNR (发射增益已隐含在 R_s 中)
SNR_coherent = SNR_lin;

theta_deg = linspace(0, 12, 121);   % 0~12 度
theta_rad = deg2rad(theta_deg);

% 对称阵列: 位置从 -(M-1)/2 到 (M-1)/2
n = -(M-1)/2 : (M-1)/2;
% 导向矢量及其导数 (θ 为弧度)
a = @(theta) exp(-1j * pi * d_lambda * n' * sin(theta));
da_dtheta = @(theta) -1j * pi * d_lambda * cos(theta) * n' .* a(theta);

% 预分配
CRB_coherent = zeros(size(theta_deg));
CRB_orth = zeros(size(theta_deg));

%% 相干信号 (R_s = a a')
for idx = 1:length(theta_rad)
    th = theta_rad(idx);
    a_vec = a(th);
    a_dot = da_dtheta(th);
    
    R_s = a_vec * a_vec';
    R_sT = R_s.';
    
    % 公式 (44)
    term1 = M * (a_dot' * R_sT * a_dot);
    term2 = (a_vec' * R_sT * a_vec) * (a_dot' * a_dot);
    term3 = M * abs(a_vec' * R_sT * a_dot)^2 / (a_vec' * R_sT * a_vec);
    
    Denom = real(term1 + term2 - term3);
    CRB_var_rad2 = 1 / (2 * SNR_coherent * Denom);
    CRB_coherent(idx) = sqrt(CRB_var_rad2) * (180/pi);
end

%% 正交信号 (R_s = I) - 使用小角度近似，CRB 为常数 (θ=0° 时的值)
% 计算 θ=0° 时的 ∥a_dot∥^2
a_vec0 = a(0);
a_dot0 = da_dtheta(0);
norm_a_dot2 = a_dot0' * a_dot0;   % = (π)^2 * sum(n^2)

% 正交信号 CRB 常数 (弧度²)
CRB_var_rad2_orth = 1 / (4 * M * SNR_orth * norm_a_dot2);
CRB_orth_const = sqrt(CRB_var_rad2_orth) * (180/pi);
CRB_orth(:) = CRB_orth_const;

%% 绘图
figure('Position', [100, 100, 600, 450]);
semilogy(theta_deg, CRB_coherent, 'b-', 'LineWidth', 2, 'DisplayName', 'Coherent signals (\beta=1)');
hold on;
semilogy(theta_deg, CRB_orth, 'r--', 'LineWidth', 2, 'DisplayName', 'Orthogonal signals (\beta=0)');
xlabel('Target direction \theta (deg)');
ylabel('CRB on DOA (deg)');
title('Figure 7: M=10, L=1, SNR=0dB');
grid on; grid minor;
legend('Location', 'best');
xlim([0, 12]);
% ylim([0, 0.35]);        % 与原图一致
hold off;




















