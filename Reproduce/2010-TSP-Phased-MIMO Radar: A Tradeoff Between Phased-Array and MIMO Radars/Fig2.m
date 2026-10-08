%% Fig. 2: Waveform diversity beampatterns using conventional beamformer
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
% D_K(theta): waveform diversity factor defined after Eq. (34).
% Fig. 2 contains D_K only, without C_K or the receive factor R.
clear;
clc;
close all;

%% Parameters: Section V and Example 1
M = 10;
K = 5;
dT = 0.5;                         % Transmit spacing in wavelengths
theta_s = 10;                     % Target direction in degrees
theta = -90:0.1:90;               % Same reproduction grid as Fig1.m

%% Phased-array radar: one waveform, D_1(theta) = 1, Eq. (36)
D_PH = ones(size(theta));

%% MIMO radar: M orthogonal waveforms, Eqs. (14), (34), and (37)
m = (0:M - 1).';
d_MIMO = exp(-1j * 2 * pi * dT * m * sind(theta));
d_MIMO_s = exp(-1j * 2 * pi * dT * m * sind(theta_s));
D_MIMO = abs(d_MIMO_s' * d_MIMO).^2 / M^2;

%% Phased-MIMO radar: K fully overlapping subarrays, Eqs. (14), (34)
% The subarray reference antennas are at positions 1, 2, ..., K.
% Hence D_K has K entries, not M - K + 1 entries.
k = (0:K - 1).';
d = exp(-1j * 2 * pi * dT * k * sind(theta));
d_s = exp(-1j * 2 * pi * dT * k * sind(theta_s));
D_PH_MIMO = abs(d_s' * d).^2 / K^2;

%% Independent check using the closed-form geometric series
delta = sind(theta) - sind(theta_s);
q = pi * dT * delta;
index_regular = abs(sin(q)) > 1e-12;
D_MIMO_check = ones(size(theta));
D_PH_MIMO_check = ones(size(theta));
D_MIMO_check(index_regular) = ...
    abs(sin(M * q(index_regular)) ./ (M * sin(q(index_regular)))).^2;
D_PH_MIMO_check(index_regular) = ...
    abs(sin(K * q(index_regular)) ./ (K * sin(q(index_regular)))).^2;

err_MIMO = max(abs(D_MIMO - D_MIMO_check));
err_PH_MIMO = max(abs(D_PH_MIMO - D_PH_MIMO_check));
[~, index_s] = min(abs(theta - theta_s));

assert(err_MIMO < 1e-12, 'MIMO formula check failed.');
assert(err_PH_MIMO < 1e-12, 'Phased-MIMO formula check failed.');
assert(all(D_PH == 1), 'Phased-array pattern must be flat.');
assert(abs(D_MIMO(index_s) - 1) < 1e-12, 'MIMO peak check failed.');
assert(abs(D_PH_MIMO(index_s) - 1) < 1e-12, ...
    'Phased-MIMO peak check failed.');

%% Convert normalized power to dB; do not square D_K again
D_PH_dB = 10 * log10(max(D_PH, realmin));
D_MIMO_dB = 10 * log10(max(D_MIMO, realmin));
D_PH_MIMO_dB = 10 * log10(max(D_PH_MIMO, realmin));

%% Fig. 2: original colors, line styles, labels, and axis limits
figure(2);
set(gcf, 'Color', 'w', 'Position', [100, 100, 860, 640]);
plot(theta, D_PH_dB, 'm--', 'LineWidth', 1.5);
hold on;
plot(theta, D_MIMO_dB, 'r:', 'LineWidth', 1.5);
plot(theta, D_PH_MIMO_dB, 'b-', 'LineWidth', 1.5);
grid on;
box on;
xlim([-90, 90]);
ylim([-80, 20]);
set(gca, 'XTick', -80:20:80, 'YTick', -80:10:20, ...
    'FontName', 'Times New Roman', 'FontSize', 18, ...
    'LineWidth', 1, 'GridLineStyle', ':');
xlabel('ANGLE (DEGREES)', 'FontName', 'Times New Roman', 'FontSize', 18);
ylabel('$|D(\theta)|^2$ (dB)', 'Interpreter', 'latex', 'FontSize', 18);
legend({'PHASED-ARRAY RADAR', 'MIMO RADAR', ...
    'PHASED-MIMO RADAR (K=5)'}, 'Location', 'northwest', ...
    'FontName', 'Times New Roman', 'FontSize', 14);
drawnow;

%% Save in the current MATLAB directory
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, 'Fig2.png', '-dpng', '-r600');
print(gcf, 'Fig2.pdf', '-dpdf', '-painters', '-bestfit');

%% Numerical verification report
fprintf('MIMO formula error: %.3e\n', err_MIMO);
fprintf('Phased-MIMO formula error: %.3e\n', err_PH_MIMO);
fprintf('Normalized peaks at theta_s = %.1f deg: %.12f, %.12f\n', ...
    theta_s, D_MIMO(index_s), D_PH_MIMO(index_s));