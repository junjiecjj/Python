clc;
clear all;
close all;
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
rng(42);


%% 参数设置
M = 10;
C = 1;
c = ones(M, 1) * C / M;
theta_est = 0;
Delta = 30;
theta_grid = -90:0.1:90;
theta_plot = theta_grid;
L = length(theta_grid);

%% 统一的期望 beampattern
P_des = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - Delta & theta_grid <= theta_est(i) + Delta;
end
P_des(idx) = 1;

%% 统一的导向矢量
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

%% =========================
%  文献1：On Probing Signal Design For MIMO Radar, C. Beampattern Matching Design
%  diag(R)=1/M, trace(R)=1, wc=0
%  =========================
w_l = ones(L, 1);
w_c = 0;
[R_lit1, alpha_lit1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);

%% =========================
%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  helperMMSECovariance 默认 diag(R)=1, trace(R)=M
%  为了和文献1对齐，将 R 除以 M，使 trace(R)=1
%  =========================
d = 0.5;
lambda = 2 * d;
pos = (0:M-1) * d;
normalizedPos = pos / lambda;
R_lit2_raw = helperMMSECovariance(normalizedPos, P_des, theta_grid);
R_lit2 = R_lit2_raw / M;

%% 计算 beampattern
P_lit1 = zeros(size(theta_plot));
P_lit2 = zeros(size(theta_plot));
for i = 1:length(theta_plot)
    ai = a(theta_plot(i));
    P_lit1(i) = real(ai' * R_lit1 * ai);
    P_lit2(i) = real(ai' * R_lit2 * ai);
end
P_lit1(P_lit1 < 0) = 0;
P_lit2(P_lit2 < 0) = 0;

%% 文献1的 alpha-scaled desired
P_des_lit1 = alpha_lit1 * P_des;

%% 为文献2单独计算一个最优 alpha，用于误差评估，不改变 R
alpha_lit2 = (P_des(:)' * P_lit2(:)) / (P_des(:)' * P_des(:));
P_des_lit2 = alpha_lit2 * P_des;

%% 误差指标
err_lit1 = norm(P_lit1(:) - P_des_lit1(:)) / norm(P_des_lit1(:));
err_lit2_to_lit1_des = norm(P_lit2(:) - P_des_lit1(:)) / norm(P_des_lit1(:));
err_lit2_self_alpha = norm(P_lit2(:) - P_des_lit2(:)) / norm(P_des_lit2(:));

fprintf('===== Power check =====\n');
fprintf('trace(R_lit1) = %.6f\n', real(trace(R_lit1)));
fprintf('trace(R_lit2) = %.6f\n', real(trace(R_lit2)));
fprintf('max diag error lit1 = %.4e\n', max(abs(real(diag(R_lit1)) - 1/M)));
fprintf('max diag error lit2 = %.4e\n', max(abs(real(diag(R_lit2)) - 1/M)));
fprintf('\n===== Alpha and error =====\n');
fprintf('alpha_lit1 = %.6f\n', alpha_lit1);
fprintf('alpha_lit2 best fit = %.6f\n', alpha_lit2);
fprintf('lit1 error to alpha_lit1 * P_des = %.4e\n', err_lit1);
fprintf('lit2 error to alpha_lit1 * P_des = %.4e\n', err_lit2_to_lit1_des);
fprintf('lit2 error to alpha_lit2 * P_des = %.4e\n', err_lit2_self_alpha);

%% 线性尺度对比
figure(1);
plot(theta_plot, P_des_lit1, 'k--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_lit1, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_lit2, 'r-.', 'LineWidth', 1.5);
grid on;
xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('Desired: \alpha_{lit1} P_d', 'Lit1: Beampattern Matching', 'Lit2: Squared Error', 'Location', 'best');
title('Comparison under trace(R)=1');
xlim([-90, 90]);

%% 绝对 dB 对比
figure(2);
plot(theta_plot, 10 * log10(P_des_lit1 + eps), 'k--', 'LineWidth', 1.5); hold on;
plot(theta_plot, 10 * log10(P_lit1 + eps), 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, 10 * log10(P_lit2 + eps), 'r-.', 'LineWidth', 1.5);
grid on;
xlabel('\theta (degrees)');
ylabel('Beampattern (dB)');
legend('Desired: \alpha_{lit1} P_d', 'Lit1: Beampattern Matching', 'Lit2: Squared Error', 'Location', 'best');
title('Absolute comparison under trace(R)=1');
xlim([-90, 90]);
ylim([-40, 5]);

%% 归一化 dB 形状对比
figure(3);
plot(theta_plot, 10 * log10(P_des + eps), 'k--', 'LineWidth', 1.5); hold on;
plot(theta_plot, 10 * log10(P_lit1 / max(P_lit1) + eps), 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plot, 10 * log10(P_lit2 / max(P_lit2) + eps), 'r-.', 'LineWidth', 1.5);
grid on;
xlabel('\theta (degrees)');
ylabel('Normalized Beampattern (dB)');
legend('Desired', 'Lit1: Beampattern Matching', 'Lit2: Squared Error', 'Location', 'best');
title('Shape comparison');
xlim([-90, 90]);
ylim([-40, 5]);

