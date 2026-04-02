

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
[R, alpha, ~ ] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);

Rsqrt =  sqrtm(R); % R^(0.5);
% 网格参数（覆盖感兴趣的区域）
theta_plot = -90:0.1:90;      % 度，网格点
Nlst = floor(logspace(1, 5, 12));
Iters = 100;

res = zeros(1, length(Nlst));

for i = 1:length(Nlst)
    N = Nlst(i);
    for it = 1:Iters
        w = (randn(M, N) + 1j * randn(M, N)) / sqrt(2);
        x = Rsqrt * w;
        Rxx = x*x'/N;
        beamdiff = zeros(size(theta_plot));

        for j = 1:length(theta_plot)
            a_theta = a(theta_plot(j));
            beamdiff(j) = abs(a_theta' * (Rxx - R) * a_theta ) / abs(a_theta' * R * a_theta);
        end
        res(i) = res(i) + mean(beamdiff);
    end
end
res = res/Iters;


%% 可选：绘制发射波束图对比
figure(1);
loglog(Nlst, res, 'k--', 'LineWidth', 1.5, 'marker','d'); hold on;
 
xlabel('Sample Number');
ylabel('MSE');
title('Transmit Beampattern');
grid on;








