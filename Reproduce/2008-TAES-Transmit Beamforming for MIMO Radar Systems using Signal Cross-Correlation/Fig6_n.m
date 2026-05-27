
 

clc;
clear all;
close all;
addpath('./functions');

rng('default');

N = 10;
d = 0.5;
lambda = 2 * d;

% ULA 一维阵元位置，单位为 wavelength
pos = ((0:N-1) - (N-1) / 2) * d;
normalizedPos = pos / lambda;

% Three targets of interest
tgtAz = [-40 0 40];                % Azimuths of the targets of interest

ang = linspace(-90, 90, 200);       % Grid of azimuth angles
beamwidth = 10;                     % Desired beamwidth

A = steeringMatrixULA1D(normalizedPos, ang);

% Desired beam pattern
Bdes = zeros(size(ang));
idx = false(size(ang));
for i = 1:numel(tgtAz)
    idx = idx | ang >= tgtAz(i) - beamwidth / 2 & ang <= tgtAz(i) + beamwidth / 2;
end
Bdes(idx) = 1;

figure(1);
plot(ang, Bdes, 'LineWidth', 2);
xlabel('Azimuth (deg)');
ylabel('Desired Beam Pattern');
title('Desired Beam Pattern');
grid on;


%% A. Squared Error Optimization
Pt = 1;
Rmmse = helperMMSECovariance(normalizedPos, Bdes, ang);
Bmmse = abs(diag(A'*Rmmse*A))/(4*pi);

%% B. Maximum Error Optimization
Rminmax = helperMinMaxCovariance(normalizedPos, Bdes, ang);
Bminmax = abs(diag(A'*Rminmax*A))/(4*pi);

%% Plot Fig
figure(2);
plot(ang, 10 * log10(Bdes / max(Bdes) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(ang, 10 * log10(Bmmse / max(Bmmse) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
plot(ang, 10 * log10(Bminmax / max(Bminmax) + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');
xlabel('Azimuth (deg)');
ylabel('Normalized (dB)');
legend('Desired', 'MMSE Covariance', 'MinMax Covariance');
ylim([-40 6]);
title('Transmit Beam Pattern');
grid on;

figure(3);
Bdes_plot = N * Bdes / (2 * pi * trapz(deg2rad(ang), Bdes .* cosd(ang)));
plot(ang, 10 * log10(Bdes_plot + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(ang, 10 * log10(Bmmse + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
plot(ang, 10 * log10(Bminmax + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');
xlabel('Azimuth (deg)');
ylabel('(dB)');
legend('Desired', 'MMSE Covariance', 'MinMax Covariance');
ylim([-40 6]);
title('Transmit Beam Pattern');
grid on;
