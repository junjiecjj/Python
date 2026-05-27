

clc;
clear all;
close all;

rng(42); 
addpath('./functions_2008TAES_CrossCorre');


%% 
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
p_des = N * P_des / (2 * pi * trapz(deg2rad(theta_grid), P_des .* cosd(theta_grid)));
 

% 问题(24)的求解, in "2008-TAES-Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation"
d = 0.5;
lambda = 2 * d;
pos = (0:N-1) * d;
normalizedPos = pos / lambda;
R_mmse = helperMMSECovariance(normalizedPos, P_des, theta_grid);

P_opt0 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_mmse * a_theta)/(4*pi);
end

rho = 1.1;

%%  Optimal R in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
L  = 256;
X_optR = WaveformSynthesisXoptimR(L, R_mmse, rho );

Rhat1 = X_optR * X_optR'/L/(4*pi);
P_opt1 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt1(i) = real(a_theta' * Rhat1 * a_theta);
end

%%  PAR < rho in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
X_par = WaveformSynthesisXwithPAR(L, R_mmse, rho  );
Rhat2 = X_par * X_par'/L/(4*pi);
P_opt2 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt2(i) = real(a_theta' * Rhat2 * a_theta);
end

%% 可选：绘制发射波束图对比
figure(1);
plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt0, 'r-', 'LineWidth', 2.5); hold on;
plot(theta_grid, P_opt1, 'b-.', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt2, 'c--', 'LineWidth', 1.5); 

xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('Desired',  'Optimized MMSE', 'CA:optimal R', 'CA:PAR = 1.1');
title('Transmit Beampattern');
grid on;
















