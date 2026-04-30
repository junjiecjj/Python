%% 图8 严格按(41)-(43)实现，适用于任意M（M=2,10,...）
% 参数：M阵元数，d_lambda=0.5，SNR=0dB，双目标，θ1=0°，θ2=5,10,15°

clc;
clear all;
close all;
rng(42); 

%% 用户可修改参数
M = 2;                          % 阵元数（可改为10，但图8原文是2）
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数（CRB与N成反比，取1即可）
alpha1 = 1; alpha2 = 1;         % 复振幅
sigma_w2 = 1 / SNR_lin;         % 噪声方差（N|α|^2/σ^2=1）

theta1_deg = 0;
theta2_list = [5, 10, 15];
beta_vec = linspace(0, 0.9999, 200);   % 相关系数

% 对称阵列（质心原点）
n = 0:M-1;         % 阵元位置（长度为M）
a = @(th) exp(-1j * 2 * pi * d_lambda * n' * sind(th));   % M×1
da = @(th) -1j * 2 * pi * d_lambda * cosd(th) * n' .* a(th); % M×1
A = @(th) a(th) * a(th).';      % M×M
dA = @(th) da(th) * a(th).' + a(th) * da(th).'; % M×M

% 相干矩阵 R_s（适用于任意M，实对称）
R_s = @(beta) (1-beta)*eye(M) + beta*ones(M);   % 标准形式

factor = 2 * N / sigma_w2;       % 公共因子 2N/σ_w^2

% 辅助函数：从复数迹构造2x2子块（对应公式(61)中的每个块）
blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];

CRB_theta1 = zeros(length(theta2_list), length(beta_vec));

for k = 1:length(theta2_list)
    theta2 = theta2_list(k);
    for b = 1:length(beta_vec)
        beta_obj = beta_vec(b);
        Rs = R_s(beta_obj);
        
        A1 = A(theta1_deg);   A2 = A(theta2);
        dA1 = dA(theta1_deg); dA2 = dA(theta2);
        
        % ---------- (60) J_θθ ----------
        T11 = trace(dA1 * Rs * dA1');
        T22 = trace(dA2 * Rs * dA2');
        T12 = trace(dA2 * Rs * dA1');   % 注意顺序：dA2 * Rs * dA1'
        Jtt = factor * [abs(alpha1)^2 * real(T11),  real(conj(alpha1)*alpha2 * T12);
                        real(conj(alpha1)*alpha2 * T12),  abs(alpha2)^2 * real(T22)];
        
        % ---------- (62) J_θa ----------
        U11 = trace(A1 * Rs * dA1');
        U22 = trace(A2 * Rs * dA2');
        U12 = trace(A2 * Rs * dA1');   % 用于θ1-α2
        U21 = trace(A1 * Rs * dA2');   % 用于θ2-α1
        
        Q11 = conj(alpha1) * U11;
        Q12 = conj(alpha2) * U12;
        Q21 = conj(alpha1) * U21;
        Q22 = conj(alpha2) * U22;
        
        row11 = factor * [real(Q11), -imag(Q11)];
        row12 = factor * [real(Q12), -imag(Q12)];
        row21 = factor * [real(Q21), -imag(Q21)];
        row22 = factor * [real(Q22), -imag(Q22)];
        Jta = [row11, row12; row21, row22];   % 2×4
        
        % ---------- (61) J_aa ----------
        S11 = trace(A1 * Rs * A1');
        S22 = trace(A2 * Rs * A2');
        S12 = trace(A2 * Rs * A1');
        Jaa = [blk(S11, factor), blk(S12, factor);
               blk(conj(S12), factor), blk(S22, factor)];   % 4×4
        
        % ---------- 稳定计算 Schur 补 ----------
        % 对 Jaa 添加正则化避免奇异（beta接近1时）
        reg = 1e-8 * trace(Jaa) / size(Jaa,1);
        Jaa_reg = Jaa + reg * eye(size(Jaa));
        Schur = Jtt - Jta * (Jaa_reg \ Jta');
        
        if rcond(Schur) > 1e-12
            CRB_theta = inv(Schur);
            CRB_deg = sqrt(CRB_theta(1,1)) * 180/pi;
        else
            CRB_deg = NaN;
        end
        CRB_theta1(k,b) = CRB_deg;
    end
end

%% 绘图
figure;
semilogy(beta_vec, CRB_theta1(1,:), 'b-', 'LineWidth',1.5); hold on;
semilogy(beta_vec, CRB_theta1(2,:), 'r--', 'LineWidth',1.5);
semilogy(beta_vec, CRB_theta1(3,:), 'g-.', 'LineWidth',1.5);
xlabel('\beta'); ylabel('CRB on DOA (deg)');
legend('\theta_2 = 5°','\theta_2 = 10°','\theta_2 = 15°');
grid on;
title(sprintf('Figure 8 (M=%d, SNR=0dB) via (60)-(62)', M));