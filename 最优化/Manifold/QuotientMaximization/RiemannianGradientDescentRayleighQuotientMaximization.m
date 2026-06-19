clc;
clear;
close all;
rng(42);
addpath('./functions');
% 构造一个 3×3 Hermitian 矩阵 A
A = [3, 1+1j, 0.5; 
    1-1j, 2, -0.3j; 
    0.5, 0.3j, 1];
A = (A + A') / 2;

% 设置算法参数
opts.eta0 = 1;
opts.rho = 0.5;
opts.c = 1e-4;
opts.epsilon = 1e-10;
opts.Kmax = 1000;
opts.max_backtrack = 100;
opts.eta_min = 1e-16;

% 随机初始化
n = size(A, 1);
opts.x0 = randn(n, 1) + 1j * randn(n, 1);

% 调用黎曼梯度下降算法
[x, lambda_est, info] = riemannian_gd_rayleigh(A, opts);

% 直接使用 eig 验证结果
[V, D] = eig(A);
lambda_all = real(diag(D));
[lambda_true, idx] = min(lambda_all);
x_true = V(:, idx);

% 复特征向量存在全局相位不唯一，需要相位对齐
phase_factor = x_true' * x;
x_aligned = x * exp(-1j * angle(phase_factor));

% 相位对齐后的误差
eigvec_err = norm(x_aligned - x_true);

% 方向一致性，越接近 1 越好
corr_val = abs(x_true' * x);

% 输出结果
fprintf('\nRiemannian GD result:\n');
fprintf('Status                              = %s\n', info.message);
fprintf('Estimated minimum Rayleigh quotient = %.12f\n', lambda_est);
fprintf('True minimum eigenvalue             = %.12f\n', lambda_true);
fprintf('Absolute eigenvalue error           = %.4e\n', abs(lambda_est - lambda_true));
fprintf('Number of iterations                = %d\n', info.iter);
fprintf('Final gradient norm                 = %.4e\n', info.grad_hist(end));
fprintf('Phase-aligned eigenvector error     = %.4e\n', eigvec_err);
fprintf('Absolute inner product              = %.12f\n', corr_val);

disp('Estimated eigenvector x:');
disp(x);

disp('True eigenvector from eig:');
disp(x_true);

disp('Estimated eigenvector after phase alignment:');
disp(x_aligned);

% 绘制瑞利商下降曲线
figure;
plot(1:info.iter, info.obj_hist, 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Rayleigh quotient');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% 绘制黎曼梯度范数收敛曲线
figure;
semilogy(1:info.iter, info.grad_hist, 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Riemannian gradient norm');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% 绘制 Armijo 步长变化
figure;
semilogy(1:info.iter, info.eta_hist, 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Step size');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% 绘制每次迭代的回溯次数
figure;
stem(1:info.iter, info.bt_hist, 'LineWidth', 1.2);
grid on;
xlabel('Iteration');
ylabel('Backtracking number');
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);