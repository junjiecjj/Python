


%% 问题(19)的SOCP求解 
clc;
clear all;
close all;

rng(42); 
addpath('./functions');

%% 1. 参数设置（示例，可修改）
M = 10;                     % 天线数
c = ones(M,1);                      % 对角元固定值
theta_est = [-40, 0, 40];   % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

Delta = 5;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i)-Delta & theta_grid <= theta_est(i)+Delta;
end
P_des(idx) = 1;
L = length(theta_grid);
 
% 权重
w_l = ones(L, 1);           % 所有网格点权重相同

wc = 1;                    % 交叉项权重（可调）
[R, alpha, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);

Rsqrt =  sqrtm(R); % R^(0.5);

N = 256;

w = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
x = Rsqrt * w;
Rxx = x*x'/N;
% 网格参数（覆盖感兴趣的区域）
theta_plot = -90:0.1:90;      % 度，网格点
beamdiff = zeros(size(theta_plot));
P_opt = zeros(size(theta_plot));
P_optxx = zeros(size(theta_plot));
for i = 1:length(theta_plot)
    a_theta = a(theta_plot(i));
    beamdiff(i) = (a_theta' * (Rxx - R) * a_theta ) / (a_theta' * R * a_theta) ;
    P_opt(i)    = (a_theta' * R * a_theta);
    P_optxx(i)  = (a_theta' * Rxx * a_theta);
end


%% 可选：绘制发射波束图对比
figure(1);
plot(theta_plot, beamdiff, 'k--', 'LineWidth', 1.5); hold on;
 
xlabel('\theta (degrees)');
ylabel('Beampattern diff');
title('Transmit Beampattern');
grid on;

figure(2);
plot(theta_plot, P_opt, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_optxx, 'r--', 'LineWidth', 1.5); hold on;
xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('P_{opt}', 'P_{opt} X');
grid on;
