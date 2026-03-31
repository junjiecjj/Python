

clc;
clear all;
close all;

rng(42); 
addpath('./functions');


%% 问题(19)的SOCP求解, in "2007-TSP-On Probing Signal Design For MIMO Radar"
N = 10;                       % 天线数
c = ones(N, 1);                % 对角元固定值
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

% 权重
w_l = ones(L, 1);           % 所有网格点权重相同
wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, N, w_l, wc, theta_est, theta_grid, P_des);
p_des = abs(P_des * alpha0+eps);

P_opt0 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
end

L  = 256;
Rho = [1, 1.1, 2];

PAR1 = zeros(length(Rho), N);

for k = 1:length(Rho)
    rho = Rho(k);

    %%  Optimal R in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
    X_optR = WaveformSynthesisXoptimR(L, R_opt0, rho );

    Rhat1 = X_optR * X_optR'/L;
    P_opt1 = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        a_theta = a(theta_grid(i));
        P_opt1(i) = real(a_theta' * Rhat1 * a_theta);
    end
    
    PAR1(k, :) = max(abs(X_optR).^2, [], 2) ./ (mean(abs(X_optR).^2, 2));

    %%  PAR < rho in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
    X_par = WaveformSynthesisXwithPAR(L, R_opt0, rho  );
    Rhat2 = X_par * X_par'/L;
    P_opt2 = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        a_theta = a(theta_grid(i));
        P_opt2(i) = real(a_theta' * Rhat2 * a_theta);
    end
    PAR2(k, :) = max(abs(X_par).^2, [], 2) ./ (mean(abs(X_par).^2, 2));
    %% 可选：绘制发射波束图对比
    figure(k);
    plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt0, 'r-', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt1, 'b--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt2, 'c--', 'LineWidth', 1.5); hold on;
 
    xlabel('\theta (degrees)');
    ylabel('Beampattern');
    legend('Desired',  'Optimized,w_c=0', 'CA:optimal R', ['CA:PAR = ', num2str(rho)]);
    title('Transmit Beampattern');
    grid on;

end 

figure(length(Rho) + 1);
plot(1:1:N, PAR1(1,:), 'r-o', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR1(2,:), 'b--*', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR1(3,:), 'c--d', 'LineWidth', 1.5); hold on;

xlabel('Index');
ylabel('PAR');
legend('CA(PAR=1):optimal R', 'CA(PAR<=1.1):optimal R', 'CA(PAR<=2):optimal R');
ylim([-1, 8]);
grid on;
 
figure(length(Rho) + 2);
plot(1:1:N, PAR2(1,:), 'r-o', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR2(2,:), 'b--*', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR2(3,:), 'c--d', 'LineWidth', 1.5); hold on;

xlabel('Index');
ylabel('PAR');
legend('CA(PAR=1):optimal R', 'CA(PAR<=1.1):optimal R', 'CA(PAR<=2):optimal R');
ylim([-1, 8]);
grid on;

























