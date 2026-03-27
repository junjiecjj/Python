%% 四种情况：理论注水解 vs fmincon 数值解（最终稳定版）
clear; clc; close all;
rng(42);

%% 通用参数
N = 4;          % 发射天线 / 目标维数
M = 6;          % 接收天线
L = N;          % 雷达快拍数（取 L=N 简化）
PT = 10;        % 总发射功率
sigma_c2 = 1;   % 高斯白噪声方差

%% 1. 雷达 MI 角度（互信息最大化）
fprintf('========== 1. 雷达 MI 角度 ==========\n');
% 生成 H 和 Z，计算协方差
H_rand = randn(M, N) + 1i*randn(M, N);
Z_rand = randn(M, L) + 1i*randn(M, L);
Sigma_H = H_rand' * H_rand / M;   % N×N
Sigma_Z = Z_rand' * Z_rand / M;   % L×L

% 特征分解（降序）
[U_H, Lambda_H] = eig(Sigma_H);
Lambda_H = real(diag(Lambda_H));
[Lambda_H, idx_H] = sort(Lambda_H, 'descend');
U_H = U_H(:, idx_H);
[U_Z, Lambda_Z] = eig(Sigma_Z);
Lambda_Z = real(diag(Lambda_Z));
[Lambda_Z, idx_Z] = sort(Lambda_Z, 'descend');
U_Z = U_Z(:, idx_Z);

% 理论解：注水公式 (1)
Lambda_H_asc = flip(Lambda_H);          % 升序
Lambda_Z_sel = Lambda_Z(end-N+1:end);   % 最小的 N 个噪声特征值（升序）
[P_mi, water_level] = water_filling(Lambda_Z_sel, Lambda_H_asc, L*PT);
fprintf('理论功率分配: %s\n', mat2str(P_mi',4));

% fmincon 数值解：优化功率分配 s（对应 Lambda_H_asc）
obj_mi = @(s) -sum(log(s .* Lambda_H_asc + Lambda_Z_sel));
Aeq = ones(1,N); beq = L*PT;
lb = zeros(N,1);
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
s0 = PT * ones(N,1);
[s_opt, fval] = fmincon(obj_mi, s0, [], [], Aeq, beq, lb, [], [], options);
P_mi_num = s_opt';

fprintf('fmincon功率分配: %s\n', mat2str(P_mi_num,4));
fprintf('绝对误差: %.2e\n', norm(P_mi - P_mi_num));

% 计算互信息（容量）
X_opt = sqrt(L) * U_Z * diag(sqrt(P_mi)) * U_H';
C_mi = log(det(X_opt * Sigma_H * X_opt' + Sigma_Z));
X_opt_num = sqrt(L) * U_Z * diag(sqrt(P_mi_num)) * U_H';
C_mi_num = log(det(X_opt_num * Sigma_H * X_opt' + Sigma_Z));
fprintf('互信息理论: %.8f\n', C_mi);
fprintf('互信息fmincon: %.8f\n\n', C_mi_num);

%% 2. 雷达估计角度（最小化 MMSE）
% 文档推导表明：最小化 MMSE 的最优功率分配与 MI 角度相同
fprintf('========== 2. 雷达估计角度 ==========\n');
% 理论功率分配与 MI 角度相同
P_mmse = P_mi;
fprintf('理论功率分配: %s\n', mat2str(P_mmse',4));

% fmincon 数值解：最小化 MMSE（对角化后，使用正确的 MMSE 表达式）
sigma_h = Lambda_H_asc;
sigma_z = Lambda_Z_sel;
% MMSE = sum( sigma_h * sigma_z ./ (s.*sigma_h + sigma_z) )
% 注意：此表达式与互信息最大化等价（通过恒等式可证明）
obj_mmse = @(s) sum(sigma_h .* sigma_z ./ (s(:).*sigma_h + sigma_z));
Aeq = ones(1,N); beq = L*PT;
lb = zeros(N,1);
[s_opt, fval] = fmincon(obj_mmse, s0, [], [], Aeq, beq, lb, [], [], options);
P_mmse_num = s_opt';

fprintf('fmincon功率分配: %s\n', mat2str(P_mmse_num,4));
fprintf('绝对误差: %.2e\n', norm(P_mmse - P_mmse_num));

% 计算 MMSE 值（两种表达式验证）
MMSE_theory = sum(sigma_h .* sigma_z ./ (P_mmse.*sigma_h + sigma_z));
MMSE_num = sum(sigma_h .* sigma_z ./ (P_mmse_num.*sigma_h + sigma_z));
fprintf('MMSE理论: %.8f\n', MMSE_theory);
fprintf('MMSE fmincon: %.8f\n', MMSE_num);
fprintf('理论解是否更优？ %s\n', string(MMSE_theory <= MMSE_num + 1e-12));
fprintf('\n');
