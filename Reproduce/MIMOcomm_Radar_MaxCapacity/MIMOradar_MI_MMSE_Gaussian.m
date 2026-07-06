

clc;
clear;
close all;

rng(1);

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultTextFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 6);
set(0, 'DefaultFigureColor', 'w');

%% ============================================================
%  MIMO Radar: MI 与 MSE 白噪声最优解一致性验证
%  Model: Y = X H + Z
%
%  X: L x N
%  H: N x M
%  Y: L x M
%  Z: L x M
%
%  Design variable:
%  S = X^H X, S is N x N
% ============================================================

L = 128;
N = 6;
M = 12;
PT = 1;
sigma_z2 = 1;

%% 构造 H 的列协方差 Sigma_H
A = randn(N, N) + 1j * randn(N, N);
Sigma_H = A * A';
Sigma_H = Sigma_H / trace(Sigma_H) * N;
Sigma_H = (Sigma_H + Sigma_H') / 2;

[U_H, D_H] = eig(Sigma_H);
lambda_H = real(diag(D_H));
lambda_H = max(lambda_H, 0);

%% 生成一个符合维度的随机 H，用于展示模型维度
G = randn(N, M) + 1j * randn(N, M);
H = sqrtm(Sigma_H) * G / sqrt(2);

fprintf('========== Model dimensions ==========\n');
fprintf('X: %d x %d\n', L, N);
fprintf('H: %d x %d\n', size(H, 1), size(H, 2));
fprintf('Y: %d x %d\n', L, M);
fprintf('S = X^H X: %d x %d\n', N, N);

%% ============================================================
%  1. MI 准则解析解
%
%  max sum log(1 + lambda_i p_i / sigma_z2)
%  s.t. sum p_i <= PT, p_i >= 0
% ============================================================

[p_MI, water_level_MI] = water_filling(sigma_z2 * ones(N, 1), lambda_H, PT);
S_MI = U_H * diag(p_MI) * U_H';
S_MI = (S_MI + S_MI') / 2;

MI_value = sum(log(1 + lambda_H .* p_MI / sigma_z2));

%% ============================================================
%  2. MSE 准则解析解
%
%  min sum (1/lambda_i + p_i/sigma_z2)^(-1)
%  s.t. sum p_i <= PT, p_i >= 0
%
%  白噪声下，其 KKT 解与 MI 有相同的注水形式
% ============================================================

[p_MSE, water_level_MSE] = water_filling(sigma_z2 * ones(N, 1), lambda_H, PT);

S_MSE = U_H * diag(p_MSE) * U_H';
S_MSE = (S_MSE + S_MSE') / 2;

MSE_value = sum(1 ./ (1 ./ lambda_H + p_MSE / sigma_z2));

%% ============================================================
%  3. 构造具体波形 X，使得 X^H X = S
% ============================================================

Q = randn(L, N) + 1j * randn(L, N);
[Q, ~] = qr(Q, 0);

X_MI = Q * diag(sqrt(p_MI)) * U_H';
X_MSE = Q * diag(sqrt(p_MSE)) * U_H';

Gram_error_MI = norm(X_MI' * X_MI - S_MI, 'fro');
Gram_error_MSE = norm(X_MSE' * X_MSE - S_MSE, 'fro');

%% ============================================================
%  4. fmincon 验证 MI 和 MSE 的功率分配
% ============================================================

options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp', 'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12, 'ConstraintTolerance', 1e-12, 'MaxFunctionEvaluations', 1e5);

Aeq = ones(1, N);
beq = PT;
lb = zeros(N, 1);
ub = PT * ones(N, 1);
p0 = PT / N * ones(N, 1);

obj_MI = @(p) -sum(log(1 + p(:) .* lambda_H(:) / sigma_z2));
[p_MI_fmincon, fval_MI] = fmincon(obj_MI, p0, [], [], Aeq, beq, lb, ub, [], options);

obj_MSE = @(p) sum(1 ./ (1 ./ lambda_H(:) + p(:) / sigma_z2));
[p_MSE_fmincon, fval_MSE] = fmincon(obj_MSE, p0, [], [], Aeq, beq, lb, ub, [], options);

S_MI_fmincon = U_H * diag(p_MI_fmincon) * U_H';
S_MI_fmincon = (S_MI_fmincon + S_MI_fmincon') / 2;

S_MSE_fmincon = U_H * diag(p_MSE_fmincon) * U_H';
S_MSE_fmincon = (S_MSE_fmincon + S_MSE_fmincon') / 2;

MI_value_fmincon = sum(log(1 + lambda_H .* p_MI_fmincon / sigma_z2));
MSE_value_fmincon = sum(1 ./ (1 ./ lambda_H + p_MSE_fmincon / sigma_z2));

%% ============================================================
%  5. 用矩阵表达式再算一遍 MI 和 MSE
% ============================================================

Sigma_half = sqrtm(Sigma_H);

MI_matrix_MI = logdet_hermitian_pd(eye(N) + Sigma_half * S_MI * Sigma_half / sigma_z2);
MI_matrix_MSE = logdet_hermitian_pd(eye(N) + Sigma_half * S_MSE * Sigma_half / sigma_z2);

MSE_matrix_MI = real(trace((Sigma_H \ eye(N) + S_MI / sigma_z2) \ eye(N)));
MSE_matrix_MSE = real(trace((Sigma_H \ eye(N) + S_MSE / sigma_z2) \ eye(N)));

%% ============================================================
%  6. 输出结果
% ============================================================

fprintf('\n========== White Noise MIMO Radar: MI vs MSE ==========\n');

fprintf('\nEigenvalues of Sigma_H:\n');
disp(lambda_H.');

fprintf('MI theoretical power allocation:\n');
disp(p_MI.');

fprintf('MSE theoretical power allocation:\n');
disp(p_MSE.');

fprintf('MI fmincon power allocation:\n');
disp(p_MI_fmincon.');

fprintf('MSE fmincon power allocation:\n');
disp(p_MSE_fmincon.');

fprintf('\nPower gap: MI theory vs MSE theory       = %.4e\n', norm(p_MI - p_MSE));
fprintf('Power gap: MI theory vs MI fmincon      = %.4e\n', norm(p_MI - p_MI_fmincon));
fprintf('Power gap: MSE theory vs MSE fmincon    = %.4e\n', norm(p_MSE - p_MSE_fmincon));
fprintf('Power gap: MI fmincon vs MSE fmincon    = %.4e\n', norm(p_MI_fmincon - p_MSE_fmincon));

fprintf('\nS gap: MI theory vs MSE theory           = %.4e\n', norm(S_MI - S_MSE, 'fro'));
fprintf('S gap: MI theory vs MI fmincon          = %.4e\n', norm(S_MI - S_MI_fmincon, 'fro'));
fprintf('S gap: MSE theory vs MSE fmincon        = %.4e\n', norm(S_MSE - S_MSE_fmincon, 'fro'));

fprintf('\nMI scalar value, theory                  = %.8f\n', MI_value);
fprintf('MI scalar value, fmincon                 = %.8f\n', MI_value_fmincon);
fprintf('\nMI matrix value using S_MI               = %.8f\n', MI_matrix_MI);
fprintf('MI matrix value using S_MSE              = %.8f\n', MI_matrix_MSE);

fprintf('MSE scalar value, theory                 = %.8f\n', MSE_value);
fprintf('MSE scalar value, fmincon                = %.8f\n', MSE_value_fmincon);
fprintf('MSE matrix value using S_MI              = %.8f\n', MSE_matrix_MI);
fprintf('MSE matrix value using S_MSE             = %.8f\n', MSE_matrix_MSE);

fprintf('\nWaveform Gram error for X_MI             = %.4e\n', Gram_error_MI);
fprintf('Waveform Gram error for X_MSE            = %.4e\n', Gram_error_MSE);

%% ============================================================
%  7. 画图
% ============================================================

plot_waterfilling(sigma_z2 ./ lambda_H, p_MI, water_level_MI);

fig = figure;
set(fig, 'Position', [100, 100, 900, 600]);

bar([p_MI(:), p_MSE(:), p_MI_fmincon(:), p_MSE_fmincon(:)], 'grouped');
xlabel('Eigenmode index');
ylabel('Power allocation');
legend('MI theory', 'MSE theory', 'MI fmincon', 'MSE fmincon', 'Location', 'best');
grid on;
box on;

%% ============================ Local functions ============================

function [power_allocation, water_level] = water_filling(sigma2, lambda, PT)
    N = length(lambda);
    noise_terms = sigma2(:) ./ lambda(:);
    [noise_sorted, idx_sorted] = sort(noise_terms, 'ascend');
    water_level = 0;
    k_active = N;
    for k = 1:N
        water_candidate = (PT + sum(noise_sorted(1:k))) / k;
        if k == N || water_candidate <= noise_sorted(k+1)
            water_level = water_candidate;
            k_active = k;
            break;
        end
    end
    power_allocation = zeros(N, 1);
    for i = 1:N
        if i <= k_active
            power_allocation(idx_sorted(i)) = max(0, water_level - noise_sorted(i));
        else
            power_allocation(idx_sorted(i)) = 0;
        end
    end
end

function plot_waterfilling(noise_powers, optimal_powers, water_level)
    N = length(noise_powers);
    x = 1:N;
    fig = figure;
    set(fig, 'Position', [100, 100, 800, 600]);
    Y = [noise_powers(:), optimal_powers(:)];
    b = bar(x, Y, 'stacked', 'BarWidth', 0.5);
    hold on;
    h_water = yline(water_level, '--', sprintf('Water level: %.3f', water_level), 'LineWidth', 2);
    xlabel('Eigenmode index');
    ylabel('Power');
    legend([b(1), b(2), h_water], {'Noise level', 'Power allocation', 'Water level'}, 'Location', 'best');
    grid on;
    box on;
end

function val = logdet_hermitian_pd(A)
    A = (A + A') / 2;
    R = chol(A);
    val = 2 * sum(log(real(diag(R))));
end