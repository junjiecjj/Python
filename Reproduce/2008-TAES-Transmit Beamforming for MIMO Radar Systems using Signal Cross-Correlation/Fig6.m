clc;
clear all;
close all;
addpath('./functions');

rng('default');

N = 10;
d = 0.5;
lambda = 2 * d;
Pt = 10;

% ULA 一维阵元位置，单位为 wavelength
pos = ((0:N-1) - (N-1) / 2) * d;
pos = (0:N-1) * d;
normalizedPos = pos / lambda;

% Targets of interest
theta_est = 0;

theta_grid = linspace(-90, 90, 200);
beamwidth = 35;

% Desired beam pattern
P_des = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - beamwidth / 2 & theta_grid <= theta_est(i) + beamwidth / 2;
end
P_des(idx) = 1;

figure(1);
plot(theta_grid, P_des, 'LineWidth', 2);
xlabel('Azimuth (deg)');
ylabel('Desired Beam Pattern');
title('Desired Beam Pattern');
grid on;

A = steeringMatrixULA1D(normalizedPos, theta_grid);
%% A. Squared Error Optimization
Rmmse = helperMMSECovariance(normalizedPos, P_des, theta_grid);
Rmmse = Rmmse * (Pt/N);
P_mmse = abs(diag(A'*Rmmse*A))/(4*pi);

fprintf('trace(Rmmse) = %.6f\n',  trace(Rmmse));

%% A. Squared Error Optimization, 不用 cos(theta) 权重，不做积分归一化，不用 barrier/Newton，直接 CVX 最小化二范数。
[Rmmse1, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt);
P_mmse1 = abs(diag(A'*Rmmse1*A))/(4*pi);
fprintf('trace(Rmmse1) = %.6f\n',  trace(Rmmse1));


%% B. Maximum Error Optimization
Rminmax = helperMinMaxCovariance(normalizedPos, P_des, theta_grid);
Rminmax = Rminmax * (Pt/N);
P_minmax = abs(diag(A'*Rminmax*A))/(4*pi);

fprintf('trace(Rminmax) = %.6f\n',  trace(Rminmax));

%% Plot Fig
figure(2);
plot(theta_grid, 10 * log10(P_des / max(P_des) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(theta_grid, 10 * log10(P_mmse / max(P_mmse) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
plot(theta_grid, 10 * log10(P_mmse1 / max(P_mmse1) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'm'); hold on;
plot(theta_grid, 10 * log10(P_minmax / max(P_minmax) + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');
xlabel('Azimuth (deg)');
ylabel('Normalized (dB)');
legend('Desired', 'MMSE Covariance', 'my MMSE', 'MinMax Covariance');
ylim([-40 5]);
title('Transmit Beam Pattern');
grid on;

figure(3);
P_des_plot = Pt * P_des / (2 * pi * trapz(deg2rad(theta_grid), P_des .* cosd(theta_grid)));
plot(theta_grid, 10 * log10(P_des_plot + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(theta_grid, 10 * log10(P_mmse + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
plot(theta_grid, 10 * log10(P_mmse1 + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'm'); hold on;
plot(theta_grid, 10 * log10(P_minmax + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');
xlabel('Azimuth (deg)');
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'my MMSE', 'MinMax Covariance');
ylim([-40 5]);
title('Transmit Beam Pattern');
grid on;