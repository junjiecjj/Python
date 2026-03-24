
 
clc;
clear all;
close all;

rng(42); 
addpath('./functions');

%% 1. 参数设置（示例，可修改）
M = 10;                     % 天线数
c = 1;                      % 对角元固定值
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
CorreCoeff = zeros(3, 7);
wc_lst = logspace((-3), 3, 7);    % 交叉项权重 
for j = 1:length(wc_lst)
    j
    wc = wc_lst(j);
    [R_opt, alpha, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);
    cross_terms = 0;
    idx = 0;
    for k=1:K-1
        for p=k+1:K
            idx = idx + 1;
            CorreCoeff(idx, j) = CorreCoeff(idx, j) + abs( a(theta_est(k))' * R_opt * a(theta_est(p)) );
        end
    end
end

%% 绘图对比
colors = colormap(jet(3));
markers = {'o', 's', '^', '*', 'd', 'v', '<', '>'};
lstys = {'-', '--', ':', '-.', };

figure(1);
for i = 1:3
    data = CorreCoeff(i, :);  % /max(CorreCoeff, [],'all');
    % data = data/max(data);
    semilogx(wc_lst, data,...
        'LineWidth', 1.5,...
        'color',colors(i,:),...
        'marker',markers(i),...
        'linestyle', lstys(i)); hold on;
end

xlabel('w_c');
ylabel('Correlation Coeffcients');
legend('1&2', '1&3', '2&3');
grid on;
