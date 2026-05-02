%% 通用程序：严格按 (41)-(43) 计算 CRB(θ)（适用于任意 M 和任意 L）
% 支持任意目标数 L，任意阵元数 M，任意相关系数矩阵 R_s（此处使用典型结构）
% 示例：L=3, M=10, SNR=0dB，固定 θ1=0°, θ2=5°, θ3=10°，扫描 β

clear; clc; close all;

%% 用户参数（可任意修改）
M = 10;                         % 阵元数
L = 3;                          % 目标数（例如 3）
d_lambda = 0.5;                 % 半波长间距
SNR_dB = 0;                     % 信噪比 (dB)
SNR_lin = 10^(SNR_dB/10);
N = 1;                          % 快拍数
alphas = ones(L, 1);            % 所有目标复振幅为1（可修改）
sigma_w2 = N * abs(alphas(1))^2 / SNR_lin;   % 噪声方差

% 角度设置（度）
theta_deg = [0, 5, 10];         % 三个目标的角度（固定，此处示例）
beta_vals = linspace(0, 0.99, 100);   % 相关系数扫描

% 对称阵列（质心在原点）
n = 0 : (M-1);         % 阵元位置
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
CRB_theta1 = zeros(1, length(beta_vals));

%% 主循环：扫描 β
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
            U = trace(A_cell{p} * Rs * dA_cell{l}');
            Q = conj(alphas(p)) * U;
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

    % 使用 pinv 替代反斜杠，更加稳健
    Schur = Jtt - Jta / Jaa * Jta';
    
    % 检查 Schur 矩阵的条件数
    if rcond(Schur) > 1e-12
        CRB_theta = inv(Schur);
        CRB_theta1_deg = CRB_theta(1,1);
    else
        CRB_theta1_deg = NaN;
    end
    CRB_theta1(b) = CRB_theta1_deg;
end

%% 绘图
figure;
semilogy(beta_vals, CRB_theta1, 'b-', 'LineWidth', 1.5);
xlabel('\beta');
ylabel('CRB on \theta_1 (deg)');
title(sprintf('CRB for L=%d, M=%d, SNR=0dB', L, M));
grid on; xlim([0,1]);