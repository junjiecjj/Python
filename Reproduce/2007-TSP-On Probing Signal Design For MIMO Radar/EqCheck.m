


clc;
clear all;
close all;
addpath('./functions');
% 设置矩阵维度
M = 3; % 可修改，但注意 J 的大小为 M^2 × M^2，M 较大时内存可能紧张

% 生成一个随机的 Hermitian 矩阵
% 先生成一个复矩阵，再取 Hermitian 部分
A = randn(M) + 1i*randn(M);
R = (A + A')/2; % 保证 Hermitian
n_diag = M;
n_upper = M*(M-1)/2;
len_r = n_diag + 2*n_upper; % 应该等于 M^2
% 构造 J 和 r
% r 的顺序：先对角线元素（实数），然后按行优先顺序的上三角元素（实部、虚部交替）
% 即对于 i=1:M-1, j=i+1:M，依次存放 real(R(i,j)) 和 imag(R(i,j))

[J, r] = generateJr(M, R);
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

% 可选：显示部分结果
disp('r 向量（前几个元素）:');
disp(r(1:min(10, len_r))');
disp('J 矩阵大小:');
disp(size(J));

























































