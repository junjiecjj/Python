




clc;
clear all;
close all;
addpath('./functions');
rng(42);

rng(1);
N = 4;
K = 3;
L = 10;
PT = 1;

H = (randn(K, N) + 1j * randn(K, N)) / sqrt(2);
data = randi([0, 3], K, L);
S = pskmod(data, 4, pi / 4, 'gray');
tmp = randn(N, L) + 1j * randn(N, L);
R = tmp * tmp';
Rd = (R + R') / 2;

X1 = strict_waveform(H, S, Rd, L);
X2 = strict_waveform1(H, S, Rd, L);


function X = strict_waveform(H, S, Rd, L)
    N = size(H, 2);
    F = chol(Rd, 'lower');
    A = F' * H' * S;
    [U, ~, V] = svd(A, 'econ');
    VN = V(:, 1:N);
    X = sqrt(L) * F * U * VN';
end


function X_opt = strict_waveform1(H, S, R_d, L)
    % 求解问题：min ||H X - S||_F^2  s.t. (1/L) X X^H = R_d
    % 输入：H - M×N, S - M×L, R_d - N×N Hermitian 正定, L (要求 L ≥ N)
    % 输出：X_opt = F * V_A * U_{A,1}^H，其中 F 来自 C = L*R_d = F*F^H
    [M, N_h] = size(H);
    N = size(R_d, 1);          % 从 R_d 获取 N
    if N_h ~= N
        error('H 的列数必须与 R_d 的维度一致');
    end
    [M_s, L_s] = size(S);
    if M_s ~= M || L_s ~= L
        error('S 的尺寸必须与 H X 匹配');
    end
    if L < N
        error('约束要求 L ≥ N，否则无解');
    end
    if ~isequal(R_d, R_d')
        error('R_d 必须是 Hermitian 矩阵');
    end
    
    C = L * R_d;
    % Cholesky 分解 C = F * F^H (F 为下三角)
    F = chol(C, 'lower');           % N×N
    H_tilde = H * F;                % M×N
    A = S' * H_tilde;               % L×N
    [U_A, ~, V_A] = svd(A, 'econ'); % U_A: L×L, V_A: N×N (因为 L≥N)
    U_A1 = U_A(:, 1:N);             % L×N
    X_opt = F * V_A * U_A1';
end




