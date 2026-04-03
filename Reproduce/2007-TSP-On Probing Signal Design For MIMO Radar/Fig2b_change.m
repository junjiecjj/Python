


%% 问题(19)的SOCP求解 
clc;
clear all;
close all;

rng(42); 
addpath('./functions');

%% 1. 参数设置（示例，可修改）
M = 10;                     % 天线数
c = ones(M,1) * 1/M;        % 对角元固定值
% c = rand(M, 1)
theta_est = [-40, 0, 40];   % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

Delta = 10;
theta_grid = -90:0.05:90;
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
[R_opt1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);
wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);

% 网格参数（覆盖感兴趣的区域）
theta_plot = -90:0.1:90;      % 度，网格点
% Delta = 10;             % Desired beamwidth
% % 期望波束图 P_des(theta)（示例：主瓣在0°，宽度约20°的sinc平方形状）
P_des = zeros(size(theta_plot));
% Desired beam pattern
idx = false(size(theta_plot));
for i = 1:numel(theta_est)
    idx = idx | theta_plot >= theta_est(i)-Delta & theta_plot <= theta_est(i)+Delta;
end
P_des(idx) = 1;

P_opt1 = zeros(size(theta_plot));
P_opt0 = zeros(size(theta_plot));
for i = 1:length(theta_plot)
    a_theta = a(theta_plot(i));
    P_opt1(i) = real(a_theta' * R_opt1 * a_theta);
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
end

%% 可选：绘制发射波束图对比
figure(1);
plot(theta_plot, abs(P_des * alpha0), 'r--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_opt1, 'r-', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_opt0, 'b--', 'LineWidth', 1.5); hold on;

% plot(theta_plot, pow2db(P_des + 1e-3), 'k--', 'LineWidth', 1.5); hold on;
% plot(theta_plot, pow2db(P_opt1/max(P_opt1)), 'r-', 'LineWidth', 1.5); hold on;
% plot(theta_plot, pow2db(P_opt0/max(P_opt0)), 'b--', 'LineWidth', 1.5); hold on;
% ylim([-30 5]);

xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('Desired', 'Optimized,w_c=1', 'Optimized,w_c=0');
title('Transmit Beampattern');
grid on;
