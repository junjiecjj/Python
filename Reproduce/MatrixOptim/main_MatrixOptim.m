

clc;
clear all;
close all;

rng(42); 


%% 主脚本：三种矩阵优化问题的求解示例, 与markdown文档对应。

% 示例 1
fprintf('===== 示例 1 =====\n');
M = 3; 
N = 5; 

S = randn(M, N) + 1i*randn(M, N);

[U, s, V] = svd(S, 'econ');
a = sum(diag(s).^2) / M;
% a = 2.0;

X1 = solve_type1(S, a);
fprintf('X X^H =\n'); disp(X1 * X1');
fprintf('目标函数值：%f\n', norm(X1 - S, 'fro')^2);
fprintf('理论最优值：%f\n', sum((sqrt(a) - svd(S)).^2));
fprintf('\n');

% 示例 2
fprintf('===== 示例 2 =====\n');
M = 4; 
N = 6; 
L = 10; 
P_T = 10.0;
H = randn(M, N) + 1i*randn(M, N);
S = randn(M, L) + 1i*randn(M, L);
X2 = solve_type2(H, S, P_T, N, L);
c = L * P_T / N;
fprintf('X X^H =\n'); disp(X2 * X2');
fprintf('应等于 c*I =\n'); disp(c * eye(N));
fprintf('目标函数值：%f\n', norm(H*X2 - S, 'fro')^2);
% 计算理论最优值
A = S' * H;
svals = svd(A);
opt_val = norm(S, 'fro')^2 + c * trace(H'*H) - 2*sqrt(c)*sum(svals(1:N));
fprintf('理论最优值：%f\n', opt_val);
fprintf('\n');

% 示例 3
fprintf('===== 示例 3 =====\n');
M = 3
N = 4; 
L = 10; 
% 构造 Hermitian 正定矩阵 R_d
R_d = randn(N, N) + 1i*randn(N, N);
R_d = R_d * R_d';   % 确保正定
H = randn(M, N) + 1i*randn(M, N);
S = randn(M, L) + 1i*randn(M, L);
X3 = solve_type3(H, S, R_d, L);
fprintf('(X X^H)/L =\n'); disp((X3 * X3') / L);
fprintf('应等于 R_d =\n'); disp(R_d);
fprintf('目标函数值：%f\n', norm(H*X3 - S, 'fro')^2);
fprintf('\n');


























































