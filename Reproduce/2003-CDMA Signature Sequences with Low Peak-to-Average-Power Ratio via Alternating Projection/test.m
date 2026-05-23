



clc;
clear all;
close all;

rng(42); 


% 测试初始化正交性
M = 9; 
N = 20;
d_lambda = 0.5;

n = 0:M-1;
n = (-(M - 1) / 2 : (M - 1) / 2).';
a_fun = @(theta) exp(-1j * 2 * pi * d_lambda * n * sin(theta));
ad_fun = @(theta) -1j * 2 * pi * d_lambda * cos(theta) * n .* a_fun(theta);

theta_target = 12 * pi/180;

a = a_fun(theta_target);
ad = ad_fun(theta_target);

% 生成随机矩阵，取 SVD 的前 M 列作为正交基
X = randn(M, N) + 1j*randn(M, N);
R = X*X'/N;



abs(a.' * R * conj(ad))

abs(ad' * R.' * a)

abs(a' * R.' * ad)