%% 复现 Bekkerman & Tabrikian (2006) Figure 7 - 严格按公式(44)实现
% M=10 阵元，ULA，半波长间距，阵列对称（质心原点）
% 相干信号: R_s = a(θ)a^H(θ) (波束指向目标方向)
% 正交信号: R_s = I, 有效 SNR = M * SNR_base (TOT补偿)
% 横轴: 目标方向θ (0~12°), 纵轴: DOA估计CRB (度)

clc;
clear all;
close all;
% addpath('./functions');

rng(42); 

%% 参数设置
M = 10;                     % 阵元数
d_lambda = 0.5;             % 半波长间距
SNR_dB = 0;                 % 基础信噪比 (0 dB)
SNR_base_lin = 10^(SNR_dB/10);   % 线性值

% 正交信号由于TOT补偿，有效SNR提高M倍
SNR_orth = M * SNR_base_lin;
% 相干信号：发射增益已隐含在R_s中，故SNR使用基础值
SNR_coherent = SNR_base_lin;

theta_deg = linspace(0, 12, 121);   % 0~12度
theta_rad = deg2rad(theta_deg);

% 对称阵列位置（质心为原点）
n = -(M-1)/2 : (M-1)/2;   % -4.5:4.5

% 导向矢量 a(θ) 及其导数 da/dθ (θ 为弧度)
a = @(theta) exp(-1j * pi * d_lambda * n' * sin(theta));
da_dtheta = @(theta) -1j * pi * d_lambda * cos(theta) * n' .* a(theta);

% 预分配
CRB_coherent = zeros(size(theta_deg));
CRB_orth = zeros(size(theta_deg));

%% 相干信号: R_s = a(θ) a^H(θ)
for idx = 1:length(theta_rad)
    th = theta_rad(idx);
    a_vec = a(th);
    a_dot = da_dtheta(th);
    
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
end

%% 正交信号: R_s = I
for idx = 1:length(theta_rad)
    th = theta_rad(idx);
    a_vec = a(th);
    a_dot = da_dtheta(th);
    
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
end

%% 绘图
figure(1);
semilogy(theta_deg, CRB_coherent, 'b-', 'LineWidth', 2, 'DisplayName', 'Coherent (\beta=1)'); hold on;
semilogy(theta_deg, CRB_orth, 'r--', 'LineWidth', 2, 'DisplayName', 'Orthogonal (\beta=0)');
xlabel('Target direction \theta (deg)');
ylabel('CRB on DOA (deg)');
title('Figure 7: M=10, L=1, SNR=0dB (strict formula (44))');
grid on; 
grid minor;
legend('Location', 'best', 'FontSize', 22);
% xlim([0, 12]);
% ylim([0, 0.35]);
hold off;