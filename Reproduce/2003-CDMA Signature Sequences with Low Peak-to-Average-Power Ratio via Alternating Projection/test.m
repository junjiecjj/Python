



clc;
clear all;
close all;

rng(42); 

% 测试初始化正交性
M = 4; N = 40;
rng(42);  % 固定种子

% 生成随机矩阵，取 SVD 的前 M 列作为正交基
X = randn(M, N) + 1j*randn(M, N);
R = X*X'/N;

Rsqrt = R^(0.5);
[L, D] = ldl(R);
R_sr = sqrt(D) * L';

A = randn(3, 4) + 1i * randn(3, 4);

[U, S, V] = svd(A);

[U1, S1, V1] = svd(A, 'econ');

% 生成随机矩阵，取 SVD 的前 M 列作为正交基
X = randn(M, N) + 1j*randn(M, N);
R = X*X'/N;

L_chol = chol(R, 'lower');