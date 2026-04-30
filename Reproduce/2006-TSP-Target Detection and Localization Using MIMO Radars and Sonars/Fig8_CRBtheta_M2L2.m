

%% 图8 严格按(41)-(43)实现（分块FIM，M=2示范，），与Fig8_eq57_M2L2.m完全一致。

clc;
clear all;
close all;
rng(42); 

%% 系统参数（可修改M，但图8固定M=2）
M = 2;                          % 阵元数
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % SNR = 0 dB
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数
alpha1 = 1; 
alpha2 = 1;         % 复振幅
sigma_w2 = N * abs(alpha1)^2 / SNR_lin;         % 噪声方差 (满足 N|α|^2/σ_w^2 = 1)

theta1_deg = 0;
theta2_list = [5, 10, 15];
beta_vec = linspace(0, 0.9999, 200);   % 相关系数

% 对称阵列（质心原点）
n = 0:M-1;         % 阵元位置
a = @(th) exp(-1j * 2 * pi * d_lambda * n' * sind(th));
da = @(th) -1j * 2 * pi * d_lambda * cosd(th) * n' .* a(th);
A = @(th) a(th) * a(th).';
dA = @(th) da(th) * a(th).' + a(th) * da(th).';

% 相干矩阵 R_s (实对称，通用形式)
R_s = @(beta) (1-beta)*eye(M) + beta*ones(M);   % 适用于任意M
factor = 2 * N / sigma_w2;       % 公共因子 2N/σ_w^2
% 辅助函数：从复数迹构建 2x2 子块（公式(61)的块）
blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];
CRB_theta1 = zeros(length(theta2_list), length(beta_vec));

for k = 1:length(theta2_list)
    theta2 = theta2_list(k);
    for b = 1:length(beta_vec)
        beta = beta_vec(b);
        Rs = R_s(beta);
        
        A1 = A(theta1_deg);   
        A2 = A(theta2);
        dA1 = dA(theta1_deg); 
        dA2 = dA(theta2);
        
        % ---------- (41): J_θθ (2x2) ----------
        T11 = trace(dA1 * Rs * dA1');
        T22 = trace(dA2 * Rs * dA2');
        T12 = trace(dA2 * Rs * dA1');   % 注意顺序符合(60)
        T21 = trace(dA1 * Rs * dA2');
        Jtt = factor * [abs(alpha1)^2 * real(T11),  real(conj(alpha1)*alpha2 * T12);
                        real(conj(alpha2)*alpha1 * T21),  abs(alpha2)^2 * real(T22)];
        
        % ---------- (43): J_θa (2x4) ----------
        Q11 = conj(alpha1) * trace(A1 * Rs * dA1');
        Q12 = conj(alpha1) * trace(A2 * Rs * dA1');
        Q21 = conj(alpha2) * trace(A1 * Rs * dA2');
        Q22 = conj(alpha2) * trace(A2 * Rs * dA2');
        
        row11 = factor * [real(Q11), -imag(Q11)];
        row12 = factor * [real(Q12), -imag(Q12)];
        row21 = factor * [real(Q21), -imag(Q21)];
        row22 = factor * [real(Q22), -imag(Q22)];
        Jta = [row11, row12; row21, row22];   % 2x4
        
        % ---------- (42): J_aa (4x4) ----------
        S11 = trace(A1 * Rs * A1');      % 实数
        S22 = trace(A2 * Rs * A2');
        S12 = trace(A2 * Rs * A1');      % 复数
        S21 = trace(A1 * Rs * A2');
        Jaa = [blk(S11, factor), blk(S12, factor);
               blk(S21, factor), blk(S22, factor)];
        
        % ---------- 稳定计算 Schur 补 ----------
        % % 关键：Jaa 可能接近奇异（尤其是 beta 接近 1），使用正则化求逆
        % reg = 1e-8 * trace(Jaa)/size(Jaa,1);   % 小正则化因子
        % Jaa_reg = Jaa + reg * eye(size(Jaa));
        % Jaa_inv = inv(Jaa);
        Schur = Jtt - Jta / Jaa * Jta';
        
        % Schur 矩阵应为 2x2 正定
        if rcond(Schur) > 1e-12
            CRB_theta = inv(Schur);
            % CRB_deg = sqrt(CRB_theta(1,1)) * 180/pi;
            CRB_deg =  CRB_theta(1,1);
        else
            CRB_deg = NaN;
        end
        CRB_theta1(k,b) = CRB_deg;
    end
end

%% 绘图
figure;
semilogy(beta_vec, CRB_theta1(1,:), 'b-', 'LineWidth',1.5); hold on;
semilogy(beta_vec, CRB_theta1(2,:), 'r--', 'LineWidth',1.5); hold on;
semilogy(beta_vec, CRB_theta1(3,:), 'g-.', 'LineWidth',1.5);
xlabel('\beta'); ylabel('CRB on DOA (deg)');
legend('\theta_2 = 5°','\theta_2 = 10°','\theta_2 = 15°');
grid on;
title('Figure 8: Strictly via (60)-(62) and Schur complement');