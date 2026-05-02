%% 严格按 (41)-(43) 实现 CRB(θ) 的通用程序（适用于任意 M 和任意 L）
% 可复现图8（设置 L=2, M=2, θ1=0°, θ2=5,10,15°, SNR=0dB）
% 使用分块 FIM (J_θθ, J_θa, J_aa) 和 Schur 补，避免完整 3L×3L 求逆

clear; clc; close all;

%% 用户参数
M = 2;                          % 阵元数（可修改）
L = 2;                          % 目标数（可修改，图8为2）
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数
alphas = ones(L, 1);            % 复振幅（可改为任意复数）
sigma_w2 = N * abs(alphas(1))^2 / SNR_lin;   % 噪声方差

% 角度设置：第一个目标固定0°，其余可变（图8中第二个目标取5,10,15°）
theta_deg = zeros(L, 1);
theta_deg(1) = 0;
theta2_list = [5, 10, 15];      % 用于图8的第二个目标角度列表
beta_vals = linspace(0, 0.9999, 200);   % 相关系数

% 对称阵列（质心在原点）
n = 0: (M-1);         % 阵元位置
a = @(th) exp(-1j * 2*pi*d_lambda * n' * sind(th));
da = @(th) -1j * 2*pi*d_lambda * cosd(th) * n' .* a(th);
A = @(th) a(th) * a(th).';
dA = @(th) da(th) * a(th).' + a(th) * da(th).';

% 相干矩阵（通用形式，适用于任意 M）
R_s = @(beta) (1-beta)*eye(M) + beta*ones(M);

factor = 2 * N / sigma_w2;       % 公共因子 2N/σ_w^2

% 辅助函数：从复数迹构造 2x2 子块（公式(61)中的块）
blk = @(t, f) f * [real(t), -imag(t); imag(t), real(t)];

% 预分配结果（存储第一个角度 θ1 的标准差，单位度）
CRB_theta1 = zeros(length(theta2_list), length(beta_vals));

%% 主循环：遍历不同的 θ2 和 β
for k = 1:length(theta2_list)
    theta_deg(2) = theta2_list(k);   % 更新第二个目标角度
    for b = 1:length(beta_vals)
        beta = beta_vals(b);
        Rs = R_s(beta);
        
        % 预先计算每个目标的 A(θ) 和 dA(θ)
        A_cell = cell(L, 1);
        dA_cell = cell(L, 1);
        for l = 1:L
            th = theta_deg(l);
            A_cell{l} = A(th);
            dA_cell{l} = dA(th);
        end
        
        % ---------- (60) 计算 J_θθ (L×L) ----------
        Jtt = zeros(L, L);
        for l = 1:L
            for p = 1:L
                T = trace(dA_cell{p} * Rs * dA_cell{l}');
                Jtt(l,p) = factor * real(conj(alphas(l)) * alphas(p) * T);
            end
        end
        
        % ---------- (62) 计算 J_θa (L × 2L) ----------
        Jta = zeros(L, 2*L);
        for l = 1:L          % 行对应角度 θ_l
            for p = 1:L      % 列对应目标 p 的复振幅
                % 迹 U_{l,p} = tr( A(θ_p) * R_s * dA^H(θ_l) )
                U = trace(A_cell{p} * Rs * dA_cell{l}');
                Q = conj(alphas(p)) * U;
                % 填充该目标 p 对应的两列（实部、虚部）
                col_base = 2*(p-1) + 1;
                Jta(l, col_base)   = factor * real(Q);
                Jta(l, col_base+1) = factor * (-imag(Q));
            end
        end
        
        % ---------- (61) 计算 J_aa (2L × 2L) ----------
        Jaa = zeros(2*L, 2*L);
        for l = 1:L
            for p = 1:L
                S = trace(A_cell{p} * Rs * A_cell{l}');
                block = blk(S, factor);
                row_base = 2*(l-1) + 1;
                col_base = 2*(p-1) + 1;
                Jaa(row_base:row_base+1, col_base:col_base+1) = block;
            end
        end
        
        % ---------- 稳定计算 Schur 补 ----------
        Schur = Jtt - Jta / Jaa * Jta';
        
        if rcond(Schur) > 1e-12
            CRB_theta = inv(Schur);
            CRB_theta1_deg = CRB_theta(1,1);
        else
            CRB_theta1_deg = NaN;
        end
        CRB_theta1(k, b) = CRB_theta1_deg;
    end
end

%% 绘图
figure;
semilogy(beta_vals, CRB_theta1(1,:), 'b-', 'LineWidth',1.5); hold on;
semilogy(beta_vals, CRB_theta1(2,:), 'r--', 'LineWidth',1.5); hold on;
semilogy(beta_vals, CRB_theta1(3,:), 'g-.', 'LineWidth',1.5);
xlabel('\beta'); ylabel('CRB on DOA (deg)');
legend('\theta_2 = 5°','\theta_2 = 10°','\theta_2 = 15°');
grid on; 
title(sprintf('Figure 8: M=%d, L=%d, SNR=0dB (via (41)-(43))', M, L));