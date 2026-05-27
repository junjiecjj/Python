clc;
clear all;
close all;
addpath('./functions');
rng(42);

%% Fig.4(c): Minimum sidelobe design with PAR < 1.2
M = 10;
L = 256;
rho = 1.1;

C = 1;
c = ones(M, 1) * C / M;

theta0 = 0;
theta1 = -10;
theta2 = 10;
theta_null = -30;
null_level_dB = -40;

Omega = [-90:0.1:-20, 20:0.1:90];
theta_plot = -90:0.1:90;

a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

%% Step 1：设计带 null 约束的 minimum sidelobe optimal R
R_opt = MinimumSidelobeDesignWithNull(c, M, theta0, theta1, theta2, Omega, theta_null, null_level_dB);

%% Step 2：CA 合成波形
X_optR = WaveformSynthesisXoptimR(L, R_opt, rho);
X_par = WaveformSynthesisXwithPAR(L, R_opt, rho);

%% Step 3：计算样本协方差矩阵
R_optR = X_optR * X_optR' / L;
R_par = X_par * X_par' / L;

%% Step 4：计算 beampattern
P_R = zeros(size(theta_plot));
P_optR = zeros(size(theta_plot));
P_par = zeros(size(theta_plot));

for i = 1:length(theta_plot)
    ai = a(theta_plot(i));
    P_R(i) = real(ai' * R_opt * ai);
    P_optR(i) = real(ai' * R_optR * ai);
    P_par(i) = real(ai' * R_par * ai);
end

P_R(P_R < 0) = 0;
P_optR(P_optR < 0) = 0;
P_par(P_par < 0) = 0;

P_R_dB = 10 * log10(P_R);
P_optR_dB = 10 * log10(P_optR);
P_par_dB = 10 * log10(P_par);

%% Step 5：输出检查量
[~, idx_null] = min(abs(theta_plot - theta_null));

err_optR = norm(R_optR - R_opt, 'fro') / norm(R_opt, 'fro');
err_par = norm(R_par - R_opt, 'fro') / norm(R_opt, 'fro');

fprintf('CA optimal R: R error = %.4e, %.4f dB\n', err_optR, 20 * log10(err_optR));
fprintf('CA PAR < %.1f: R error = %.4e, %.4f dB\n', rho, err_par, 20 * log10(err_par));

par_optR = L * max(abs(X_optR).^2, [], 2) ./ sum(abs(X_optR).^2, 2);
par_par = L * max(abs(X_par).^2, [], 2) ./ sum(abs(X_par).^2, 2);

fprintf('CA optimal R max PAR = %.4f\n', max(par_optR));
fprintf('CA PAR < %.1f max PAR = %.4f\n', rho, max(par_par));

fprintf('Optimal R null at %.1f° = %.4f dB\n', theta_null, P_R_dB(idx_null));
fprintf('CA optimal R null at %.1f° = %.4f dB\n', theta_null, P_optR_dB(idx_null));
fprintf('CA PAR < %.1f null at %.1f° = %.4f dB\n', rho, theta_null, P_par_dB(idx_null));

%% Step 6：画 Fig.4(c)
figure;
plot(theta_plot, P_R_dB, 'k--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_optR_dB, 'b-.', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_par_dB, 'r-', 'LineWidth', 1.5); hold on;
xline(theta_null, 'k:', 'LineWidth', 1);

grid on;
xlabel('\theta (degrees)');
ylabel('Normalized Power (dB)');
legend('Optimal R', 'CA: optimal R', 'CA: PAR < 1.2', 'Location', 'best');
title('Fig. 4(c): Minimum Sidelobe Design, PAR < 1.2');
xlim([-90, 90]);
% ylim([-60, 5]);