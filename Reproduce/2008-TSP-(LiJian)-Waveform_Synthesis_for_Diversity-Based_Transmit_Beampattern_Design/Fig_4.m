
clc;
clear all;
close all;

rng(42); 
addpath('./functions');


%% 问题(19)的SOCP求解, in "2007-TSP-On Probing Signal Design For MIMO Radar"
N = 10;                       % 天线数
c = ones(N, 1)/N;                % 对角元固定值
theta_est = [0];   % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:N-1)' * sind(theta));  % M×1

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