% This Matlab script can be used to generate the range-RMSE versus sensing-SNR
% results in the paper:
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "MIMO-OFDM ISAC waveform design for range-Doppler sidelobe suppression," IEEE Trans. Wireless Commun., vol. 24, no. 2, pp. 1001-1015, Feb. 2025.
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "Low range-Doppler sidelobe ISAC waveform design: A low-complexity approach," IEEE Trans. Veh. Technol., vol. 74, no. 10, pp. 16544-16549, Oct. 2025.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-26

clear; clc; close all;
rng('shuffle')

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));

R = load(fullfile(root,'data','waveform_result_20260326_110413.mat'), ...
    'para', 'x_comm', 'combined', 'ISAC_almrcg', 'ISAC_mmadmm', 'radar');

ITER = 2e4;             % Monte Carlo runs.
snr_db = -25:2:5;       % Weak-target sensing SNR sweep.

% Basic parameters
Ns = R.para.Ns;
Nc = R.para.Nc;
Nt = R.para.Nt;
Nr = R.para.Nr;
fc = R.para.fc;
c0 = R.para.c0;
lambda = R.para.lambda;
Tsym = R.para.Tsym;
sigma2_radar = R.para.sigma_r2;
delay_res = 1 / R.para.bandwidth;
range_res = c0 * delay_res / 2;

theta_rad = R.para.theta0_rad;
aT = exp(1j*pi*(0:Nt-1).' * sin(theta_rad));
aR = exp(1j*pi*(0:Nr-1).' * sin(theta_rad));

% Waveforms to be compared
wave_name_list = {'ISAC_mmadmm', 'ISAC_almrcg', 'radar', 'combined (RF)', ...
    'comm_only (RF)',  'comm_only'};
wave_label_list = {'MM-ADMM', 'ALM-RCG', 'Radar-only', 'Combined (RF)', ...
    'Comm-only (RF)', 'Comm-only'};
nWave = numel(wave_name_list);
range_MSE_iter = zeros(nWave, numel(snr_db), ITER);

% Precompute beamformed waveforms
Xbar_all = cell(nWave, 1);
for k = 1:nWave
    name = wave_name_list{k};
    switch name
        case 'ISAC_mmadmm'
            x = R.ISAC_mmadmm.x;
        case 'ISAC_almrcg'
            x = R.ISAC_almrcg.x;
        case 'combined (RF)'
            x = R.combined.x;
        case 'radar'
            x = R.radar.x;
        case {'comm_only', 'comm_only (RF)'}
            x = R.x_comm;
    end
    X = reshape(x, Nc * Nt, Ns);
    Xbar_all{k} = kron(eye(Nc), aT') * X;
end

Kt = 2;
SNRdB_t1 = 35;
alpha1_abs = sqrt(sigma2_radar * 10^(SNRdB_t1/10));
t1_delay_bin = 1;
t1_dopp_bin  = 0;
t1_row = mod(floor(Nc/2) + t1_delay_bin, Nc) + 1;
t1_col = mod(floor(Ns/2) + t1_dopp_bin, Ns) + 1;

for ii = 1:numel(snr_db)
    fprintf('weak target sensing SNR = %.2f dB\n', snr_db(ii));
    alpha2_abs = sqrt(sigma2_radar * 10^(snr_db(ii)/10));
    alpha_abs = [alpha1_abs; alpha2_abs];
    parfor iter = 1:ITER
        target_delay_bin = [1, 1+14*rand(1)];
        target_doppler_bin = [0, -8+7*rand(1)];
        target_delay_true = target_delay_bin*delay_res;
        target_range_true = target_delay_bin*range_res;
        alpha = alpha_abs .* exp(-1j * 2*pi * fc * target_delay_true(:));
        tarPhase = zeros(Nc, Ns, Kt);
        for it = 1:Kt
            delay_vec   = exp(-1j * 2*pi * target_delay_bin(it)   * (0:Nc-1).' / Nc);
            doppler_vec = exp(-1j * 2*pi * target_doppler_bin(it) * (0:Ns-1).' / Ns);
            tarPhase(:, :, it) = delay_vec * doppler_vec';
        end
        noise = sqrt(sigma2_radar/2) * (randn(Nr, Nc, Ns) + 1j * randn(Nr, Nc, Ns));
        N_noise = reshape(noise, Nc * Nr, Ns);
        Ynoise  = kron(eye(Nc), aR') * N_noise / Nr;

        range_err_local = zeros(nWave, 1);
        for k = 1:nWave
            Xbar = Xbar_all{k};
            Yecho = Ynoise;
            for it = 1:Kt
                Yecho = Yecho + alpha(it) * tarPhase(:, :, it) .* Xbar;
            end
            if (strcmp(wave_name_list{k}, 'comm_only (RF)')) || strcmp(wave_name_list{k}, 'combined (RF)')
                den = Xbar;
                den(abs(den) < 1e-12) = 1e-12;
                H_sensing = Yecho ./ den;
            else
                H_sensing = Yecho .* conj(Xbar);
            end
            Z_rdm = abs(fftshift(fft(ifft(H_sensing, Nc, 1), Ns, 2), 2));
            Z_rdm = fftshift(Z_rdm, 1);

            Z_rdm(t1_row, t1_col) = -inf;
            [~, idx_t2] = max(Z_rdm(:));
            [row_idx, ~] = ind2sub(size(Z_rdm), idx_t2);

            est_delay_bin = row_idx - (floor(Nc/2) + 1);
            est_range_m  = est_delay_bin * range_res;
            range_err_local(k) = (est_range_m  - target_range_true(2))^2;
        end
        range_MSE_iter(:, ii, iter) = range_err_local;
    end
end
range_RMSE_all = sqrt(mean(range_MSE_iter, 3));
%% plot
color_list = R.para.colors;
marker_list = R.para.markers;

figure('Color', 'w');
for k = 1:nWave
    semilogy(snr_db, range_RMSE_all(k, :), 'LineWidth', 1.5, ...
        'Color', color_list{k}, 'Marker', marker_list{k}, 'MarkerSize', 8);
    hold on;
end
hold off;
grid on;
xlabel('Sensing SNR (dB)', 'FontSize', 12);
ylabel('Range RMSE (m)', 'FontSize', 12);
xlim([snr_db(1), snr_db(end)]);
ylim([1e-1 50]);
legend(wave_label_list, 'FontSize', 10, 'Position', ...
    [0.130000002927013 0.108750003618855 0.287499994145972 0.258333326095627]);