% This Matlab script can be used to generate the probability-of-detection
% versus weak-target-RCS results in the paper:
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "MIMO-OFDM ISAC waveform design for range-Doppler sidelobe suppression," IEEE Trans. Wireless Commun., vol. 24, no. 2, pp. 1001-1015, Feb. 2025.
% P. Li, M. Li, R. Liu, Q. Liu, and A. L. Swindlehurst, "Low range-Doppler sidelobe ISAC waveform design: A low-complexity approach," IEEE Trans. Veh. Technol., vol. 74, no. 10, pp. 16544-16549, Oct. 2025.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-26

clear; clc; close all;
rng('shuffle')

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));

R = load(fullfile(root,'data','waveform_result_20260326_110413.mat'), ...
    'para', 'x_comm', 'combined', 'ISAC_almrcg', 'ISAC_mmadmm', 'radar');
% Basic parameters
ITER = 2e4;                     % Monte Carlo runs.
rcs1_db = 20;                   % Strong target RCS.
rcs2_db = linspace(-20, 0, 11); % Weak target RCS sweep.
Ns = R.para.Ns;
Nc = R.para.Nc;
Nt = R.para.Nt;
Nr = R.para.Nr;
fc = R.para.fc;
c0 = R.para.c0;
sigma2_radar = R.para.sigma_r2;
lambda = R.para.lambda;
delay_res = 1 / R.para.bandwidth;
range_res = c0 * delay_res / 2;

% Waveforms to be compared
wave_name_list = {'ISAC_mmadmm', 'ISAC_almrcg', 'radar', 'combined (RF)', ...
    'comm_only (RF)',  'comm_only'};
wave_label_list = {'MM-ADMM', 'ALM-RCG', 'Radar-only', 'Combined (RF)', ...
    'Comm-only (RF)', 'Comm-only'};
nWave = numel(wave_name_list);
Pd_all = zeros(nWave, numel(rcs2_db));

% Two-target setup
tarPara = struct();
tarPara.theta_rad   = R.para.theta0_rad;
tarPara.delay_bin   = [1 5];
tarPara.doppler_bin = [0 2];
tarPara.Kt          = numel(tarPara.delay_bin);
tarPara.delay_s = tarPara.delay_bin * delay_res;
tarPara.range_m = tarPara.delay_bin * range_res;

% Steering vectors
aT = exp(1j*pi*(0:Nt-1).' * sin(tarPara.theta_rad));
aR = exp(1j*pi*(0:Nr-1).' * sin(tarPara.theta_rad));

% Precompute the beamformed waveform
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

% Index of the weak target in the RDM matrix
weak_row = mod(floor(Nc/2) + tarPara.delay_bin(2), Nc) + 1;
weak_col = mod(floor(Ns/2) + tarPara.doppler_bin(2), Ns) + 1;

% Precompute delay-Doppler phase terms.
tarPhase = cell(tarPara.Kt, 1);
for it = 1:tarPara.Kt
    delay_vec   = exp(-1j * 2*pi * tarPara.delay_bin(it)   * (0:Nc-1).' / Nc);
    doppler_vec = exp(-1j * 2*pi * tarPara.doppler_bin(it) * (0:Ns-1).' / Ns);
    tarPhase{it} = delay_vec * doppler_vec';
end

for ii = 1:numel(rcs2_db)
    fprintf('weak target RCS = %.2f dBsm\n', rcs2_db(ii));
    rcs_db = [rcs1_db, rcs2_db(ii)];
    rcs    = 10.^(rcs_db / 10);
    alpha = sqrt(rcs .* lambda^2 ./ ((4*pi)^3 .* tarPara.range_m.^4)) ...
        .* exp(-1j * 2*pi * fc * tarPara.delay_s);
    hit_mat = zeros(nWave, ITER);
    parfor iter = 1:ITER
        noise = sqrt(sigma2_radar/2) * (randn(Nr, Nc, Ns) + 1j*randn(Nr, Nc, Ns));
        N_noise = reshape(noise, Nc * Nr, Ns);
        Ynoise  = kron(eye(Nc), aR') * N_noise / Nr;
        hit_local = zeros(nWave, 1);
        for k = 1:nWave
            Xbar = Xbar_all{k};
            Yecho = zeros(Nc, Ns);
            for it = 1:tarPara.Kt
                Yecho = Yecho + alpha(it) * tarPhase{it} .* Xbar;
            end
            Yecho = Yecho + Ynoise;
            if (strcmp(wave_name_list{k}, 'comm_only (RF)')) || strcmp(wave_name_list{k}, 'combined (RF)')
                den = Xbar;
                den(abs(den) < 1e-12) = 1e-12;
                H_sensing = Yecho ./ den;
            else
                H_sensing = Yecho .* conj(Xbar);
            end
            Z_rdm = abs(fftshift(fft(ifft(H_sensing, Nc, 1), Ns, 2), 2));
            Z_rdm = fftshift(Z_rdm, 1);

            detMap = zeros(size(Z_rdm));
            [~, idx] = maxk(Z_rdm(:), 2);
            detMap(idx) = 1;
            if detMap(weak_row, weak_col) == 1
                hit_local(k) = 1;
            end
        end
        hit_mat(:, iter) = hit_local;
    end
    Pd_all(:, ii) = sum(hit_mat, 2) / ITER;
end
%% plot
color_list = R.para.colors;
marker_list = R.para.markers;
figure('Color', 'w');
for k = 1:nWave
    plot(rcs2_db, Pd_all(k, :), 'LineWidth', 1.5, 'Color', color_list{k}, ...
        'Marker', marker_list{k}, 'MarkerSize', 8);
    hold on;
end
hold off;
grid on;
xlabel('Weak target RCS (dBsm)', 'FontSize', 12);
ylabel('Probability of detection', 'FontSize', 12);
xlim([rcs2_db(1), rcs2_db(end)]);
ylim([0, 1]);
legend(wave_label_list, 'FontSize', 12, 'Position', ...
    [0.563095244694322 0.200793658244233 0.312499993400914 0.264285706835134])