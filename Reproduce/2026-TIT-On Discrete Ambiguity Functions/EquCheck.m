

clc;
clear;
close all;
rng(42);

clc;
clear;
close all;


%% Eq.(14)
N = 8;
Ncp = 3;
k = 2;

if k > Ncp
    error('公式(14)要求 k <= Ncp，否则CP长度不足。');
end

Acp = [zeros(Ncp, N - Ncp), eye(Ncp); eye(N)];
Rcp = [zeros(N, Ncp), eye(N)];

L = N + Ncp;
Jtilde = zeros(L, L);
Jtilde(k + 1:L, 1:L - k) = eye(L - k);

J = zeros(N, N);
J(1:k, N - k + 1:N) = eye(k);
J(k + 1:N, 1:N - k) = eye(N - k);

Left = Rcp * Jtilde * Acp;
Right = J;

err_matrix = norm(Left - Right, 'fro');

fprintf('矩阵级验证误差 ||Left - Right||_F = %.3e\n', err_matrix);

x = randn(N, 1) + 1j * randn(N, 1);

x_left = Rcp * Jtilde * Acp * x;
x_right = J * x;

err_signal = norm(x_left - x_right);

fprintf('信号级验证误差 ||x_left - x_right||_2 = %.3e\n', err_signal);

disp('原始信号 x:');
disp(x.');

disp('加CP -> 线性时延 -> 去CP 后的结果:');
disp(x_left.');

disp('直接周期时延 J*x 的结果:');
disp(x_right.');



%% 



















