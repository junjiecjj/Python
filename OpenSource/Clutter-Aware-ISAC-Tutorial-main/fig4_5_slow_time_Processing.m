% This Matlab script can be used to generate Fig. 4 and Fig. 5 in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear;
clc;
close all;
rng('shuffle');

colors = {'#0072BD', '#77AC30', '#7E2F8E', '#D95319', '#E67DAF', '#EDB120', '#A21E2D'};
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','parameters_basic.mat'), 'para', 'tarPara', 'hotSourcePara', 'cfg')
%% Basic parameters
N_grid = 1024;                  % Angular grid size for MUSIC spectrum
Nt = para.Nt;                   % Number of transmit antennas
Nr = para.Nr;                   % Number of receive antennas
N = para.N;                     % Number of subcarriers
L = para.L;                     % Number of OFDM symbols
Ku = para.Ku;                   % Number of communication users
Ns = para.Ns;                   % Number of transmit streams
Pt = para.Pt;                   % Total transmit power
deltaf = para.deltaf;           % Subcarrier spacing
fc = para.fc;                   % Carrier frequency
QAM_order = para.QAM_order;     % Communication modulation order
Nt_hs = hotSourcePara.Nt_hs;    % Number of hot-source antennas
dist_range = (0:N-1) * para.range_res;         % Range axis
velo_range = (-L/2:L/2-1) * para.velo_res;     % Velocity axis

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);

% Generate clutter channels
[H_cold, H_hot, H_uav, uavPara] = gen_clutter_channels(para, hotSourcePara, tarPara, cfg);
%% Generate beamformers and transmit signals
H_comm = (randn(Nt, N, Ku) + 1j * randn(Nt, N, Ku)) / sqrt(2);   % Communication channels
W_total = zeros(Nt, Ns, N);    % Overall beamformer
Ws = zeros(Nt, Ns-Ku, N);      % Sensing beamformer
Wc = zeros(Nt, Ku, N);         % Communication beamformer
for n = 0:N-1
    for u = 1:Ku
        Wc(:, u, n+1) = H_comm(:, n+1, u) / norm(H_comm(:, n+1, u));
    end
    for s = 1:Ns-Ku
        target_idx = mod(s-1, tarPara.M) + 1;
        Ws(:, s, n+1) = a_n(tarPara.theta(target_idx), n);
    end
    W_total(:, :, n+1) = [Wc(:, :, n+1), Ws(:, :, n+1)];
end
W_total = W_total * sqrt(Pt / norm(W_total, 'fro')^2);

% Communication symbols
bits_comm = randi([0, QAM_order-1], Ku, N);
s_comm = qammod(bits_comm, QAM_order, 'UnitAveragePower', true);
% Sensing symbols
bits_sens = randi([0, 3], Ns-Ku, N);
s_sens = qammod(bits_sens, 4, 'UnitAveragePower', true);
% Combined transmit symbols, repeated across OFDM symbols
s_total = [s_comm; s_sens];
s_total = repmat(s_total, [1, 1, L]);

% Transmit signal
x_nl = zeros(Nt, N, L);
for n = 1:N
    x_nl(:, n, :) = W_total(:, :, n) * squeeze(s_total(:, n, :));
end

% Hot-source transmitted waveform
x_hs = sqrt(hotSourcePara.ILpower / N / Nt_hs / 2) * ...
    (randn(Nt_hs, N, L) + 1j * randn(Nt_hs, N, L));
%% Generate echoes and interference components
y_tar = echo_tar(para, tarPara, x_nl, a_n, b_n);   % Target echo
y_uav = zeros(Nr, N, L);                           % UAV return
y_cold = zeros(Nr, N, L);                          % Cold clutter
y_hot = zeros(Nr, N, L);                           % Hot clutter
for l = 1:L
    for n = 1:N
        y_uav(:, n, l) = H_uav(:, :, l, n) * x_nl(:, n, l);
        y_cold(:, n, l) = H_cold(:, :, l, n) * x_nl(:, n, l);
        y_hot(:, n, l) = H_hot(:, :, l, n) * x_hs(:, n, l);
    end
end
noise = sqrt(para.sigma2 / 2) * ...
    (randn(Nr, N, L) + 1j * randn(Nr, N, L));      % Receiver noise
%% Scenario 1: cold clutter only
y_nl_s1 = y_tar + y_uav + y_cold + noise;

% slow-time domain processing
[y_subtracted_s1, ~] = symbol_wise_mean_esti(y_nl_s1);  %Avg.
RMApara.lambda = 0.85;
RMApara.initialization = 'first_symbol';
[y_rma_s1, ~] = rma_filter(y_nl_s1, RMApara);           %RMA
y_diff_s1 = frame_wise_difference(y_nl_s1);             %CSD

% MUSIC spectra for Scenario 1
Ms = 10;
theta_music = linspace(-90, 90, N_grid);
Pmusic_s1 = music_spectrum(y_nl_s1, Ms, theta_music);
Pmusic_ave_s1 = music_spectrum(y_subtracted_s1, Ms, theta_music);
Pmusic_rma_s1 = music_spectrum(y_rma_s1, Ms, theta_music);
Pmusic_diff_s1 = music_spectrum(y_diff_s1, Ms, theta_music);
Pmusic_nordB_s1 = 10 * log10(Pmusic_s1 / max(Pmusic_s1));
Pmusic_ave_nordB_s1 = 10 * log10(Pmusic_ave_s1 / max(Pmusic_ave_s1));
Pmusic_rma_nordB_s1 = 10 * log10(Pmusic_rma_s1 / max(Pmusic_rma_s1));
Pmusic_diff_nordB_s1 = 10 * log10(Pmusic_diff_s1 / max(Pmusic_diff_s1));

p_music_s1 = figure('Color', 'white', 'Units', 'centimeters', 'Position', [5, 5, 16, 12]);
plot_music = plot(theta_music, Pmusic_nordB_s1, 'color', colors{1}, 'LineWidth', 1.5);
hold on;
plot_ave = plot(theta_music, Pmusic_ave_nordB_s1, 'color', colors{2}, 'LineWidth', 1.5);
plot_diff = plot(theta_music, Pmusic_diff_nordB_s1, 'color', colors{4}, 'LineWidth', 1.5);
plot_ram = plot(theta_music, Pmusic_rma_nordB_s1, 'color', colors{3}, 'LineWidth', 1.5, 'LineStyle', '-');
plot([tarPara.theta, tarPara.theta], [-150, 0], 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
plot([uavPara.thetas(1), uavPara.thetas(1)], [-150, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
plot([uavPara.thetas(2), uavPara.thetas(2)], [-150, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('Angle (°)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
grid on;
xlim([-90, 90]);
ylim([-65, 0]);
legend([plot_music, plot_ave, plot_ram, plot_diff], ...
    {'W/o Suppression', 'Avg.', 'RMA', 'CSD'}, ...
    'FontSize', 12, 'Location', 'northoutside', 'NumColumns', 4);

% RDM for Scenario 1
y_beamformed = zeros(N, L);
y_beamformed_rma = zeros(N, L);
for l = 0:L-1
    for n = 0:N-1
        w_mrc = b_n(tarPara.theta, n);
        y_beamformed(n+1, l+1) = w_mrc' * squeeze(y_nl_s1(:, n+1, l+1));
        y_beamformed_rma(n+1, l+1) = w_mrc' * squeeze(y_rma_s1(:, n+1, l+1));
    end
end
A_tx = a_n(tarPara.theta(1), 0:N-1);
A_tx_expanded = repmat(A_tx, [1, 1, L]);
s_nl = squeeze(sum(conj(A_tx_expanded) .* x_nl, 1));
H_hat_s1 = y_beamformed .* conj(s_nl);
RDM_s1 = abs(fftshift(fft(ifft(H_hat_s1, N, 1), L, 2), 2)).^2;
H_hat_rma_s1 = y_beamformed_rma .* conj(s_nl);
RDM_rma_s1 = abs(fftshift(fft(ifft(H_hat_rma_s1, N, 1), L, 2), 2)).^2;

% Plot RDMs for Scenario 1
zmax = max(max(RDM_s1(:)), max(RDM_rma_s1(:)));
zmin = -100;
p_rdm_s1 = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_s1 / zmax));
hold on;
annotation('rectangle', [0.574214285714286 0.495238095238096 0.0257857142857143 0.0333333333333341], 'LineWidth', 1);
annotation('ellipse', [0.704571428571427 0.707142857142864 0.0382857142857143 0.0309523809523814], 'LineWidth', 1);
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-100 100]);
ylim([0 200]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;

p_rdm_rma_s1 = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_rma_s1 / zmax));
hold on;
annotation('rectangle', [0.574214285714286 0.495238095238096 0.0257857142857143 0.0333333333333341], 'LineWidth', 1);
annotation('ellipse', [0.704571428571427 0.707142857142864 0.0382857142857143 0.0309523809523814], 'LineWidth', 1);
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-100 100]);
ylim([0 200]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;
%% Scenario 2: mixed clutter
y_nl_s2 = y_tar + y_uav + y_cold + y_hot + noise;
[y_subtracted_s2, ~] = symbol_wise_mean_esti(y_nl_s2);
[y_rma_s2, ~] = rma_filter(y_nl_s2, RMApara);
y_diff_s2 = frame_wise_difference(y_nl_s2);

% MUSIC spectra for Scenario 2
Ms = 12;
Pmusic_s2 = music_spectrum(y_nl_s2, Ms, theta_music);
Pmusic_ave_s2 = music_spectrum(y_subtracted_s2, Ms, theta_music);
Pmusic_rma_s2 = music_spectrum(y_rma_s2, Ms, theta_music);
Pmusic_diff_s2 = music_spectrum(y_diff_s2, Ms, theta_music);
Pmusic_nordB_s2 = 10 * log10(Pmusic_s2 / max(Pmusic_s2));
Pmusic_ave_nordB_s2 = 10 * log10(Pmusic_ave_s2 / max(Pmusic_ave_s2));
Pmusic_rma_nordB_s2 = 10 * log10(Pmusic_rma_s2 / max(Pmusic_rma_s2));
Pmusic_diff_nordB_s2 = 10 * log10(Pmusic_diff_s2 / max(Pmusic_diff_s2));

p_music_s2 = figure('Color', 'white', 'Units', 'centimeters', 'Position', [5, 5, 16, 12]);
plot_music = plot(theta_music, Pmusic_nordB_s2, 'color', colors{1}, 'LineWidth', 1.5);
hold on;
plot_ave = plot(theta_music, Pmusic_ave_nordB_s2, 'color', colors{2}, 'LineWidth', 1.5);
plot_diff = plot(theta_music, Pmusic_diff_nordB_s2, 'color', colors{4}, 'LineWidth', 1.5);
plot_ram = plot(theta_music, Pmusic_rma_nordB_s2, 'color', colors{3}, 'LineWidth', 1.5, 'LineStyle', '-');
p_tar = plot([tarPara.theta, tarPara.theta], [-150, 0], 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
plot([uavPara.thetas(1), uavPara.thetas(1)], [-150, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
plot([uavPara.thetas(2), uavPara.thetas(2)], [-150, 0], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
p_hot = plot([hotSourcePara.thetas, hotSourcePara.thetas], [-150, 0], 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
xlabel('Angle (°)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12); 
grid on;
xlim([-90, 90]);
ylim([-55, 0]);
legend([plot_music, plot_ave, plot_ram, plot_diff], ...
    {'W/o Suppression', 'Avg.', 'RMA', 'CSD'}, ...
    'FontSize', 12, 'Location', 'northoutside', 'NumColumns', 4);

% RDM for Scenario 2
y_beamformed = zeros(N, L);
y_beamformed_rma = zeros(N, L);
for l = 0:L-1
    for n = 0:N-1
        w_mrc = b_n(tarPara.theta, n);
        y_beamformed(n+1, l+1) = w_mrc' * squeeze(y_nl_s2(:, n+1, l+1));
        y_beamformed_rma(n+1, l+1) = w_mrc' * squeeze(y_rma_s2(:, n+1, l+1));
    end
end
A_tx = a_n(tarPara.theta(1), 0:N-1);
A_tx_expanded = repmat(A_tx, [1, 1, L]);
s_nl = squeeze(sum(conj(A_tx_expanded) .* x_nl, 1));
H_hat_s2 = y_beamformed .* conj(s_nl);
RDM_s2 = abs(fftshift(fft(ifft(H_hat_s2, N, 1), L, 2), 2)).^2;
H_hat_rma_s2 = y_beamformed_rma .* conj(s_nl);
RDM_rma_s2 = abs(fftshift(fft(ifft(H_hat_rma_s2, N, 1), L, 2), 2)).^2;

% Plot RDMs for Scenario 2
zmax = max(max(RDM_s2(:)), max(RDM_rma_s2(:)));
zmin = -100;
p_rdm_s2 = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_s2 / zmax));
hold on;
annotation('rectangle', [0.574214285714286 0.502380952380953 0.0257857142857143 0.0333333333333341], 'LineWidth', 1);
annotation('ellipse', [0.704571428571427 0.711904761904768 0.0382857142857143 0.0309523809523814], 'LineWidth', 1);
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-100 100]);
ylim([0 200]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;

p_rdm_rma_s2 = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_rma_s2 / zmax));
hold on;
annotation('rectangle', [0.574214285714286 0.495238095238096 0.0257857142857143 0.0333333333333341], 'LineWidth', 1);
annotation('ellipse', [0.704571428571427 0.711904761904768 0.0382857142857143 0.0309523809523814], 'LineWidth', 1);
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-100 100]);
ylim([0 200]);
zlim([zmin, 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;