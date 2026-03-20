


clc;
clear all;
close all;

rng(42); 
addpath('./functions');


%% Eq.(20)
% 设置矩阵维度
M = 3; % 可修改，但注意 J 的大小为 M^2 × M^2，M 较大时内存可能紧张
% 生成一个随机的 Hermitian 矩阵
% 先生成一个复矩阵，再取 Hermitian 部分
A = randn(M) + 1i*randn(M);
R = (A + A')/2; % 保证 Hermitian

% R = [1 3+4j; 3-4j 2];
% R = [1 4+5j 6+7j; 4-5j 2 8+9j; 6-7j 8-9j 3];

[J, r] = generateJr(size(R,1), R);
Rhat = recover_R_from_r(r, M);
% 验证 vec(R) = J * r
vec_R = R(:); % 列堆叠
J_r = J * r;

% 检查是否相等（允许数值误差）
if norm(vec_R - J_r) < 1e-10
    disp('验证成功：vec(R) = J * r');
else
    disp('验证失败，请检查代码');
    norm_diff = norm(vec_R - J_r);
    disp(['差值范数: ', num2str(norm_diff)]);
end

%% Eq.(21)(22)
% 延续之前的代码，假设已经生成 M, R, J, r, vec_R
% 若未运行，请先运行前面构造 J 和 r 的部分

% ========== 定义导向矢量函数 ==========
% 假设为均匀线阵，阵元间距半波长，角度 θ（度数）对应导向矢量
% 也可以使用任意复数矢量，这里为了通用，随机生成导向矢量
% 为验证等式，我们生成随机复数导向矢量 a1 和 a2
theta_deg = 30;
d_lambda = 0.5; % 阵元间距半波长
a1 = exp(1i * 2*pi * d_lambda * (0:M-1)' * sind(30));
a2 = exp(1i * 2*pi * d_lambda * (0:M-1)' * sind(45));
% 对于 (21)，两个导向矢量相同，即 μ_l 对应的导向矢量
a_mu = a1;   % 用于 (21)
a_theta_k = a1; % 用于 (22) 的第一个导向矢量
a_theta_p = a2; % 用于 (22) 的第二个导向矢量

% ========== 验证 (21) ==========
left21 = a_mu' * R * a_mu;
kronecker_row = kron(a_mu.', a_mu'); 
right21 = kronecker_row * J * r;

% 比较
if norm(left21 - right21) < 1e-10
    disp('(21) 验证成功：a*(μ_l) R a(μ_l) = [a^T(μ_l) ⊗ a*(μ_l)] J r');
else
    disp('(21) 验证失败');
    fprintf('差值: %e\n', norm(left21 - right21));
end

g_l = - (kronecker_row * J).'; % 列向量
right21 = -g_l.' * r;
% 比较
if norm(left21 - right21) < 1e-10
    disp('(21) 验证成功：a*(μ_l) R a(μ_l) = [a^T(μ_l) ⊗ a*(μ_l)] J r');
else
    disp('(21) 验证失败');
    fprintf('差值: %e\n', norm(left21 - right21));
end
% ========== 验证 (22) ==========
% 左侧：a*(θ_k) * R * a(θ_p)
left22 = a_theta_k' * R * a_theta_p;
kronecker_row22 = kron(a_theta_p.', a_theta_k');
right22 = kronecker_row22 * J * r;
if norm(left22 - right22) < 1e-10
    disp('(22) 验证成功：a*(θ_k) R a(θ_p) = [a^T(θ_p) ⊗ a*(θ_k)] J r');
else
    disp('(22) 验证失败');
    fprintf('差值: %e\n', norm(left22 - right22));
end

% 对于 (22)，定义 d_{k,p}^* r = 该表达式，即 d_{k,p} = conj(kronecker_row22 * J)'
d_kp = (kronecker_row22 * J)'; % 列向量
right22 = d_kp' * r;
if norm(left22 - right22) < 1e-10
    disp('(22) 验证成功：a*(θ_k) R a(θ_p) = [a^T(θ_p) ⊗ a*(θ_k)] J r');
else
    disp('(22) 验证失败');
    fprintf('差值: %e\n', norm(left22 - right22));
end





















































