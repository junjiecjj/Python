



clc;
clear all;
close all;

rng(42);  

M = 10;
c = ones(M, 1) * 1 / M;
theta_est = [ 0 ];
Delta = 10;
theta_grid = -90:0.05:90;

P_rect = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - Delta & theta_grid <= theta_est(i) + Delta;
end
P_rect(idx) = 1;

Pt = 1;
[P_des, Rd_des, alpha_des] = smoothDesiredPatternByCovariance(theta_grid, P_rect, M, Pt, 'uniform');

figure;
plot(theta_grid, P_rect, 'k--', 'LineWidth', 1.2);
hold on;
plot(theta_grid, P_des, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\theta (degrees)');
ylabel('Desired Pattern');
legend('Original rectangular', 'Smoothed desired');
title('Smoothed Desired Beampattern');


function [P_smooth, Rd, alpha_scale] = smoothDesiredPatternByCovariance(theta_grid, P_rect, M, Pt, constraint_type)
    % 输入：
    %   theta_grid      - 角度网格，单位 degree
    %   P_rect          - 原始矩形期望方向图
    %   M               - 天线数
    %   Pt              - 总功率
    %   constraint_type - 'trace' 或 'uniform'
    % 输出：
    %   P_smooth        - 由优化协方差矩阵生成的平滑方向图，已归一化到最大值为 1
    %   Rd              - 优化得到的协方差矩阵
    %   alpha_scale     - 矩形方向图的最优缩放系数
    if nargin < 5
        constraint_type = 'trace';
    end
    theta_grid = theta_grid(:).';
    P_rect = P_rect(:).';
    L = length(theta_grid);
    a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));
    cvx_begin quiet sdp
        variable Rd(M, M) hermitian semidefinite
        variable alpha_scale nonnegative
        expression u(L)
        for l = 1:L
            al = a(theta_grid(l));
            u(l) = alpha_scale * P_rect(l) - real(al' * Rd * al);
        end
        minimize norm(u, 2)
        subject to
            trace(Rd) == Pt;
            if strcmpi(constraint_type, 'uniform')
                for m = 1:M
                    real(Rd(m, m)) == Pt / M;
                end
            end
    cvx_end
    P_smooth = zeros(size(theta_grid));
    for l = 1:L
        al = a(theta_grid(l));
        P_smooth(l) = real(al' * Rd * al);
    end
    P_smooth(P_smooth < 0) = 0;
    if max(P_smooth) > 0
        P_smooth = P_smooth / max(P_smooth);
    end
end
