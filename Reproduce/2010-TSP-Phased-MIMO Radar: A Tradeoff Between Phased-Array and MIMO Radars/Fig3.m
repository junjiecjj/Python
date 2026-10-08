%% Fig. 3: Overall beampatterns using conventional transmit/receive beamformer
% Hassanien and Vorobyov, IEEE TSP, vol. 58, no. 6, June 2010.
% DOI: 10.1109/TSP.2010.2043976
% Eq. (35): G_K(theta) = C_K(theta) * D_K(theta) * R(theta).
clear;
clc;
close all;

%% Parameters: Section V and Example 1
M = 10;
N = 10;
K = 5;
dT = 0.5;                         % Transmit spacing in wavelengths
dR = 0.5;                         % Receive spacing in wavelengths
theta_s = 10;                     % Target direction in degrees
theta = -90:0.1:90;               % Same reproduction grid as Fig1.m
Mk = M - K + 1;

%% Full transmit and receive steering vectors: Eq. (8)
m = (0:M - 1).';
n = (0:N - 1).';
a = exp(-1j * 2 * pi * dT * m * sind(theta));
a_s = exp(-1j * 2 * pi * dT * m * sind(theta_s));
b = exp(-1j * 2 * pi * dR * n * sind(theta));
b_s = exp(-1j * 2 * pi * dR * n * sind(theta_s));

%% Common receive factor: definition after Eq. (34)
R = abs(b_s' * b).^2 / N^2;

%% Phased-array radar: K = 1, Eq. (36)
C_PH = abs(a_s' * a).^2 / M^2;
D_PH = ones(size(theta));
G_PH = C_PH .* D_PH .* R;

%% MIMO radar: K = M, Eq. (37)
C_MIMO = ones(size(theta));
D_MIMO = abs(a_s' * a).^2 / M^2;
G_MIMO = C_MIMO .* D_MIMO .* R;

%% Phased-MIMO transmit factor: Mk = 6 antennas per subarray
mk = (0:Mk - 1).';
a_k = exp(-1j * 2 * pi * dT * mk * sind(theta));
a_k_s = exp(-1j * 2 * pi * dT * mk * sind(theta_s));
w_k = a_k_s / norm(a_k_s);
C_PH_MIMO = abs(a_k_s' * a_k).^2 / Mk^2;

%% Phased-MIMO waveform diversity factor: K = 5 reference positions
k = (0:K - 1).';
d = exp(-1j * 2 * pi * dT * k * sind(theta));
d_s = exp(-1j * 2 * pi * dT * k * sind(theta_s));
D_PH_MIMO = abs(d_s' * d).^2 / K^2;
G_PH_MIMO = C_PH_MIMO .* D_PH_MIMO .* R;

%% Independent check using the virtual steering vector: Eqs. (18), (27)
% For equal subarrays, all entries of c(theta) have the same value.
c = ones(K, 1) * (w_k' * a_k);
c_s = ones(K, 1) * (w_k' * a_k_s);
u_s = kron(c_s .* d_s, b_s);
G_virtual = zeros(size(theta));

for index = 1:numel(theta)
    u = kron(c(:, index) .* d(:, index), b(:, index));
    G_virtual(index) = abs(u_s' * u)^2 / norm(u_s)^4;
end

err_virtual = max(abs(G_PH_MIMO - G_virtual));
err_PH_MIMO_equal = max(abs(G_PH - G_MIMO));
[~, index_s] = min(abs(theta - theta_s));

assert(err_virtual < 1e-12, 'Virtual-array formula check failed.');
assert(err_PH_MIMO_equal < 1e-12, ...
    'Phased-array and MIMO patterns must coincide.');
assert(abs(G_PH(index_s) - 1) < 1e-12, 'Phased-array peak check failed.');
assert(abs(G_MIMO(index_s) - 1) < 1e-12, 'MIMO peak check failed.');
assert(abs(G_PH_MIMO(index_s) - 1) < 1e-12, ...
    'Phased-MIMO peak check failed.');

%% Convert normalized power to dB; do not square G_K again
G_PH_dB = 10 * log10(max(G_PH, realmin));
G_MIMO_dB = 10 * log10(max(G_MIMO, realmin));
G_PH_MIMO_dB = 10 * log10(max(G_PH_MIMO, realmin));

%% Fig. 3: original colors, line styles, labels, and axis limits
figure(3);
set(gcf, 'Color', 'w', 'Position', [100, 100, 860, 640]);
plot(theta, G_PH_dB, 'm--', 'LineWidth', 1.5);
hold on;
plot(theta, G_MIMO_dB, 'r:', 'LineWidth', 1.5);
plot(theta, G_PH_MIMO_dB, 'b-', 'LineWidth', 1.5);
grid on;
box on;
xlim([-90, 90]);
ylim([-120, 30]);
set(gca, 'XTick', -80:20:80, 'YTick', -120:20:20, ...
    'FontName', 'Times New Roman', 'FontSize', 18, ...
    'LineWidth', 1, 'GridLineStyle', ':');
xlabel('ANGLE (DEGREES)', 'FontName', 'Times New Roman', 'FontSize', 18);
ylabel('$|G(\theta)|^2$ (dB)', 'Interpreter', 'latex', 'FontSize', 18);
legend({'PHASED-ARRAY RADAR', 'MIMO RADAR', ...
    'PHASED-MIMO RADAR (K=5)'}, 'Location', 'northwest', ...
    'FontName', 'Times New Roman', 'FontSize', 14);
drawnow;

%% Save in the current MATLAB directory
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, 'Fig3.png', '-dpng', '-r600');
print(gcf, 'Fig3.pdf', '-dpdf', '-painters', '-bestfit');

%% Numerical verification report
fprintf('Virtual-array formula error: %.3e\n', err_virtual);
fprintf('Phased-array/MIMO pattern difference: %.3e\n', err_PH_MIMO_equal);
fprintf('Normalized peaks at theta_s = %.1f deg: %.12f, %.12f, %.12f\n', ...
    theta_s, G_PH(index_s), G_MIMO(index_s), G_PH_MIMO(index_s));