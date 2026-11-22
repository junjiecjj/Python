% This Matlab script can be used to generate Fig. 2 in the paper:
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
%% Basic system parameters
Nt = para.Nt;                  % Number of transmit antennas
Nr = para.Nr;                  % Number of receive antennas
N = para.N;                    % Number of subcarriers
L = para.L;                    % Number of OFDM symbols
Ku = para.Ku;                  % Number of communication users
Ns = para.Ns;                  % Number of transmit streams
Pt = para.Pt;                  % Total transmit power
deltaf = para.deltaf;          % Subcarrier spacing
fc = para.fc;                  % Carrier frequency
QAM_order = para.QAM_order;    % QAM order for communication symbols
Nt_hs = hotSourcePara.Nt_hs;   % Number of hot-source antennas

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;   % Normalized frequency factor
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) ...
    * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);   % Transmit steering vector
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) ...
    * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);   % Receive steering vector

% Generate clutter channels
[H_cold, H_hot, H_uav] = gen_clutter_channels(para, hotSourcePara, tarPara, cfg);
%% Generate beamformers and transmit symbols
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
bits_comm = randi([0, QAM_order-1], Ku, N, L);
s_comm = qammod(bits_comm, QAM_order, 'UnitAveragePower', true);
% Sensing symbols
bits_sens = randi([0, 3], Ns-Ku, N, L);
s_sens = qammod(bits_sens, 4, 'UnitAveragePower', true);
% Combined transmit symbols
s_total = [s_comm; s_sens];

% Transmit signal
x_nl = zeros(Nt, N, L);
for n = 1:N
    x_nl(:, n, :) = W_total(:, :, n) * squeeze(s_total(:, n, :));
end

% Hot-source interference waveform
x_hs = sqrt(hotSourcePara.ILpower / N / Nt_hs / 2) * ...
    (randn(Nt_hs, N, L) + 1j * randn(Nt_hs, N, L));
%% Generate the received signal
y_tar = echo_tar(para, tarPara, x_nl, a_n, b_n);   % Target echo
y_clutter = zeros(Nr, N, L);                       % Clutter return
for l = 1:L
    for n = 1:N
        y_clutter(:, n, l) = (H_uav(:,:,l,n) + H_cold(:,:,l,n)) * x_nl(:, n, l)...
            + H_hot(:,:,l,n) * x_hs(:, n, l);
    end
end
noise = sqrt(para.sigma2 / 2) * ...
    (randn(Nr, N, L) + 1j * randn(Nr, N, L));      % Receiver noise
y_nl = y_tar + y_clutter + noise;                  % Received signal
%% Compute the RDM
dist_range = (0:N-1) * para.range_res;             % Range axis
velo_range = (-L/2:L/2-1) * para.velo_res;         % Velocity axis
% Receive beamforming toward the target direction
y_beamformed = zeros(N, L);
for l = 0:L-1
    for n = 0:N-1
        y_nl_slice = squeeze(y_nl(:, n+1, l+1));
        w_mrc = b_n(tarPara.theta(1), n);          % MRC beamformer
        y_beamformed(n+1, l+1) = w_mrc' * y_nl_slice;
    end
end
% Transmit-side matched-filter reference
A_tx = a_n(tarPara.theta(1), 0:N-1);
A_tx_expanded = repmat(A_tx, [1, 1, L]);
s_nl = squeeze(sum(conj(A_tx_expanded) .* x_nl, 1));
% RDM with matched filtering
H_hat_mf = y_beamformed .* conj(s_nl);
RDM_mf = abs(fftshift(fft(ifft(H_hat_mf, N, 1), L, 2), 2)).^2;
% RDM without de-randomization
RDM_woDerand = abs(fftshift(fft(ifft(y_beamformed, N, 1), L, 2), 2)).^2;
%% Plot
zmin = -100;
h_mf = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_mf / max(RDM_mf(:))));
hold on;
annotation('rectangle', [0.574214285714286 0.502380952380956 0.0257857142857143 0.0333333333333341], 'LineWidth', 1);
annotation('ellipse', [0.704571428571427 0.709523809523814 0.0382857142857143 0.0309523809523814], 'LineWidth', 1);
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-100 100]);
ylim([0 200]);
zlim([zmin 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;

h_woderand = figure('Color', 'white');
s1 = mesh(velo_range, dist_range, 10 * log10(RDM_woDerand / max(RDM_woDerand(:))));
hold on;
xlabel('Velocity (m/s)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('Range (m)', 'FontSize', 12, 'FontName', 'Times New Roman');
zlabel('Magnitude (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
view([-50, 40]);
xlim([-60 60]);
ylim([0 200]);
zlim([zmin 0]);
clim([zmin 0]);
colormap('jet');
s1.FaceColor = 'interp';
shading interp;