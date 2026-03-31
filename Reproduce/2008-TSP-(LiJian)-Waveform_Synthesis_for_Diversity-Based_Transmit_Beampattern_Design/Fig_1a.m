

clc;
clear all;
close all;

rng(42); 
addpath('./functions');


%% 问题(19)的SOCP求解 
M = 10;                     % 天线数
c = ones(M,1);                      % 对角元固定值
theta_est = [0];   % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

Delta = 30;
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

wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);


P_opt0 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
end

p_des = abs(P_des * alpha0+eps);

%% 可选：绘制发射波束图对比
figure(1);
plot(theta_grid, p_des, 'r--', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt0, 'b-', 'LineWidth', 1.5); hold on;

% plot(theta_grid, pow2db(p_des/max(p_des+eps)), 'r--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, pow2db(P_opt0/max(P_opt0)), 'b-', 'LineWidth', 1.5); hold on;
% ylim([-30 2]);

xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('Desired',  'Optimized,w_c=0');
title('Transmit Beampattern');
grid on;








































































