
%% 主程序
clear; clc; close all;

% 设置随机种子
rng(42);

%% 对齐特征向量函数
function [Psi_c_aligned, reorder] = align_eigenvectors(U_c, Psi_c)
    % 通过计算列向量间的内积来对齐U_c和Psi_c的特征向量顺序
    % 参数:
    %   U_c: Σc的特征向量矩阵
    %   Psi_c: Rc的特征向量矩阵
    % 返回:
    %   Psi_c_aligned: 对齐后的Psi_c
    %   mapping: 列映射关系

    N = size(U_c, 2);
    % 计算相关系数矩阵
    correlation_matrix = zeros(N, N);

    for i = 1:N
        for j = 1:N
            % 计算归一化内积 (a·b*)/(|a|·|b|)
            a = U_c(:, i);
            b = Psi_c(:, j);
            correlation = abs(dot(a, b)) / (norm(a) * norm(b));
            correlation_matrix(i, j) = correlation;
        end
    end

    [~, reorder] = max(correlation_matrix, [], 2);
    Psi_c_aligned = Psi_c(:, reorder);
end

%% 注水算法函数
function [optimal_powers, water_level] = water_filling(noise_var, eigenvalues, total_power)
    % 注水算法
    % 参数:
    %   noise_var: 噪声方差
    %   eigenvalues: 特征值数组
    %   total_power: 总功率
    % 返回:
    %   optimal_powers: 最优功率分配
    %   water_level: 注水水平

    % 按特征值降序排序
    [eig_vals, idx] = sort(eigenvalues, 'descend');
    N = length(eig_vals);
    optimal_powers = zeros(1, N);

    % 计算注水水平
    for i = 1:N
        water_level = (total_power + sum(noise_var ./ eig_vals(1:i))) / i;
        powers = water_level - noise_var ./ eig_vals(1:i);
        if all(powers >= 0)
            % 如果所有功率非负，则继续
            continue;
        else
            % 否则，减少一个子信道
            water_level = (total_power + sum(noise_var ./ eig_vals(1:i-1))) / (i-1);
            powers = water_level - noise_var ./ eig_vals(1:i-1);
            optimal_powers(idx(1:i-1)) = powers;
            break;
        end
    end

    % 如果所有子信道都可用，则分配功率
    if all(optimal_powers >= 0)
        optimal_powers = water_level - noise_var ./ eig_vals;
        optimal_powers(optimal_powers < 0) = 0;
        optimal_powers = optimal_powers(idx); % 恢复原始顺序
    end
end

%% 生成PSD矩阵函数
function H = generate_psd_hermitian_method1(n, seed)
    % 生成Hermitian半正定矩阵
    if nargin > 1
        rng(seed);
    end

    % 生成随机复数矩阵
    A = randn(n) + 1j * randn(n);
    % 生成Hermitian半正定矩阵
    H = A * A';
end

%% 验证Eq.(7)-(11)
fprintf('=== 验证Eq.(7)-(11) ===\n');

M = 4;
T = 100;
N = 4;
PT = 1;
sigma_c2 = 1;
I_N = eye(N);

% 生成复数信道矩阵
Hc = randn(M, N) + 1j * randn(M, N);
Sigma_C = Hc' * Hc;

% 使用CVX求解优化问题
cvx_begin
    variable Rc(N, N) complex hermitian
    maximize( log_det(Hc * Rc * Hc' / sigma_c2 + I_N) )
    subject to
        Rc == hermitian_semidefinite(N)
        trace(Rc) <= PT
cvx_end

if strcmp(cvx_status, 'Solved')
    fprintf('最优值: %.4f\n', cvx_optval);
    fprintf('Rc矩阵:\n');
    disp(Rc);
end

% 特征值分解
Lambda_c_hat = eig(Sigma_C);
Lambda_c_hat = abs(Lambda_c_hat);
[U_c, ~] = eig(Sigma_C);

% Rc的特征值分解
Lambda_c2 = eig(Rc);
Lambda_c2 = abs(Lambda_c2);
[Psi_c, ~] = eig(Rc);

% 对齐特征向量
[Psi_c_aligned, reorder] = align_eigenvectors(U_c, Psi_c);
Lambda_C2 = Lambda_c2(reorder);

% 使用注水算法
[optimal_powers, water_level] = water_filling(sigma_c2, Lambda_c_hat, PT);
fprintf('最优功率分配: %s\n', mat2str(optimal_powers));
fprintf('实际使用功率: %.4f\n', sum(optimal_powers));

% 绘制注水算法结果（可选）
figure;
subplot(1,1,1);
bar(optimal_powers);
hold on;
plot([0, N+1], [water_level, water_level], 'r--', 'LineWidth', 2);
xlabel('子信道');
ylabel('功率分配');
title('注水算法功率分配');
legend('分配功率', '注水水平');
grid on;

fprintf('Lambda_C2 = %s\n', mat2str(Lambda_C2'));
fprintf('optimal_powers = %s\n', mat2str(optimal_powers));

%% 验证Eq.(12) - 按列展开
fprintf('\n=== 验证Eq.(12) - 按列展开 ===\n');

M = 2;
N = 3;
T = 4;

% 生成复数矩阵
Hs = randn(M, N) + 1j * randn(M, N);
Xs = randn(N, T) + 1j * randn(N, T);
Ys = Hs * Xs;

% 按列展开
ys1 = reshape(Ys.', [], 1);  % 等价于Python的 Ys.conj().T.flatten('F')
I = eye(M);
Xhat = kron(I, Xs');
hs = reshape(Hs.', [], 1);   % 等价于Python的 Hs.conj().T.flatten('F')

yhat = Xhat * hs;

% 验证是否相等
fprintf('ys1和yhat的差异: %e\n', norm(ys1 - yhat));

%% 验证Eq.(12)-(16) - 相关系数矩阵
fprintf('\n=== 验证Eq.(12)-(16) - 相关系数矩阵 ===\n');

C = [1, 0.5, 0.3; 0.5, 1, 0.3; 0.3, 0.3, 1];

L = chol(C, 'lower');
U = L';

R = randn(100000, 3);
Rc = R * U;

X = Rc(:, 1);
Y = Rc(:, 2);
Z = Rc(:, 3);

C_hat = corr(Rc);
fprintf('相关系数矩阵:\n');
disp(C_hat);

%% 验证Eq.(16) - 逆向验证
fprintf('\n=== 验证Eq.(16) - 逆向验证 ===\n');

M = 4;
T = 100;
N = 4;
PT = 1;
sigma_s2 = 1;

Sigma_S = generate_psd_hermitian_method1(N, 42);
% 取实部（为了与CVX兼容）
Sigma_S = real(Sigma_S);
Lambda_s = eig(Sigma_S);
Lambda_s = abs(Lambda_s);
[U_s, ~] = eig(Sigma_S);

[Gamma_s, water_level] = water_filling(sigma_s2, abs(Lambda_s), PT);

Rs = U_s * diag(Gamma_s) * U_s';
fprintf('通过注水算法计算的Rs:\n');
disp(Rs);

%% 使用CVX求解Eq.(16)
fprintf('\n=== 使用CVX求解Eq.(16) ===\n');

Sigma_s_inv = inv(Sigma_S);

cvx_begin
    variable gamma(N) nonnegative
    minimize( sum(inv_pos(gamma/sigma_s2 + 1./Lambda_s)) )
    subject to
        sum(gamma) <= PT
cvx_end

if strcmp(cvx_status, 'Solved')
    fprintf('CVX求解的gamma: %s\n', mat2str(gamma'));
    fprintf('注水算法的Gamma_s: %s\n', mat2str(Gamma_s));
end

Rs1 = U_s * diag(gamma) * U_s';
fprintf('通过CVX计算的Rs1:\n');
disp(Rs1);

%% 直接验证Eq.(16) - 使用CVX求解原始问题
fprintf('\n=== 直接验证Eq.(16) - 使用CVX求解原始问题 ===\n');

cvx_begin
    variable RS(N, N) complex hermitian
    minimize( trace_inv(RS/sigma_s2 + Sigma_s_inv) )
    subject to
        RS == hermitian_semidefinite(N)
        trace(RS) <= PT
cvx_end

if strcmp(cvx_status, 'Solved')
    fprintf('CVX求解的RS:\n');
    disp(RS);
    fprintf('注水算法计算的Rs:\n');
    disp(Rs);
end

%% 如果是实数矩阵的情况
fprintf('\n=== 实数矩阵情况 ===\n');

cvx_begin
    variable RS_real(N, N) semidefinite
    minimize( trace_inv(RS_real/sigma_s2 + Sigma_s_inv) )
    subject to
        trace(RS_real) <= PT
cvx_end

if strcmp(cvx_status, 'Solved')
    fprintf('实数情况CVX求解的RS_real:\n');
    disp(RS_real);
end

fprintf('所有验证完成！\n');
