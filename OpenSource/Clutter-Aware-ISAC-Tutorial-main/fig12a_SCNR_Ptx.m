% This Matlab script script can be used to generate Fig. 12a in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear all;
clc;
close all;
rng('shuffle');

colors = {'#0072BD', '#77AC30', '#7E2F8E', '#D95319', '#E67DAF', '#EDB120', '#A21E2D'};
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','covMat_BLP.mat'), 'para', 'Vcc', 'Vcc_hat', 'R_eta')
%% Basic system parameters
ITER = 100;          % Number of Monte Carlo trials
N = 32;              % Number of subcarriers
L = 64;              % Number of OFDM symbols / slow-time snapshots
para.N = N;
para.L = L;
Nt = para.Nt;        % Number of transmit antennas
Nr = para.Nr;        % Number of receive antennas
fc = para.fc;        % Carrier frequency
deltaf = para.deltaf;% Subcarrier spacing
Ku = 6;              % Number of communication users
para.Ku = Ku;
% Communication settings
gamma_comm = 10^(10/10);      % Communication SINR threshold
gamma = gamma_comm * ones(N, Ku);
alpha_path = 3.1;             % Path-loss exponent
Path_loss = @(d) -30 - alpha_path * 10 * log10(d);
para.PSD_comm = 10^(-160/10) / 1e3;   % Communication noise PSD
sigma2_comm = para.PSD_comm * deltaf; % Communication noise power

% Optimization settings
opts = struct;
opts.maxAO = 30;           % Maximum AO iterations
opts.maxDinkel = 30;       % Maximum Dinkelbach iterations
opts.tolAO_dB = 0.01;      % AO stopping tolerance in dB
opts.tolDinkel_eta = 1e-3; % Dinkelbach stopping tolerance
opts.lambdaRidge = 0;      % Optional ridge regularization
opts.verbose = 0;          % Verbosity flag
f_n = para.fc + (0:para.N-1) * para.deltaf;   % Frequency grid

% Target parameters
tarPara.M = 1;                         % Number of targets
tarPara.rcs = 10.^([5] / 10);          % Target RCS
tarPara.theta = [-10]';                % Target angle (deg)
tarPara.ranges = [41.8]';              % Target range (m)
tarPara.velos = [-31.2]';              % Target radial velocity (m/s)
tarPara.delays = 2 * tarPara.ranges / para.c0;
tarPara.dopplers = 2 * tarPara.velos / para.lambda_wave;
tarPara.alpha_mn = zeros(tarPara.M, para.N);
Gtx = 10^(20/10);
Grx = 10^(20/10);
for m = 1:tarPara.M
    tarPara.alpha_mn(m, :) = sqrt(para.Pt * (para.c0 ./ f_n).^2 * Gtx * Grx ...
        * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
end
% Frequency-dependent steering vectors
chi_n = @(n) 1 + n * deltaf / fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)' * chi_n(n)) ./ sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)' * chi_n(n)) ./ sqrt(Nr);

% Build target steering vectors over subcarriers
a_t = cell(N, 1);
b_t = cell(N, 1);
for n = 0:N-1
    a_t{n+1} = a_n(tarPara.theta, n);
    b_t{n+1} = b_n(tarPara.theta, n);
end

% Transmit power settings
Ptx_dBm_range = 30:2:50;                       % Transmit power in dBm
Ptx_range_W = 10.^((Ptx_dBm_range - 30) / 10); % Convert dBm to W

% Preallocate SCNR results
SCNR_ISAC_perf_ave = zeros(size(Ptx_range_W));
SCNR_ISAC_esti_ave = zeros(size(Ptx_range_W));
SCNR_radar_perf_ave = zeros(size(Ptx_range_W));
SCNR_radar_esti_ave = zeros(size(Ptx_range_W));
SCNR_heur_perf_ave = zeros(size(Ptx_range_W));
SCNR_heur_esti_ave = zeros(size(Ptx_range_W));
SCNR_comm_ave = zeros(size(Ptx_range_W));
%% Monte Carlo evaluation versus transmit power
for ii = 1:length(Ptx_range_W)
    Ptot = Ptx_range_W(ii);
    para.Pt = Ptot;
    % Update the target response under the current transmit power
    tarPara.alpha_mn = zeros(tarPara.M, N);
    for m = 1:tarPara.M
        tarPara.alpha_mn(m, :) = sqrt(para.Pt * Gtx * Grx * (para.c0 ./ f_n).^2 ...
            * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
    end
    sigma2_tar = abs(tarPara.alpha_mn).^2;

    fprintf('Ptx = %.2f W (%.1f dBm)\n', Ptot, Ptx_dBm_range(ii));
    scnr_isac_perf_sum = 0;
    scnr_isac_esti_sum = 0;
    scnr_radar_perf_sum = 0;
    scnr_radar_esti_sum = 0;
    scnr_heur_perf_sum = 0;
    scnr_heur_esti_sum = 0;
    scnr_comm_sum = 0;

    parfor it = 1:ITER
        % Generate communication channels
        H_comm = (randn(Nt, N, Ku) + 1j * randn(Nt, N, Ku)) / sqrt(2);
        distance_Ku = randi([100, 500], 1, Ku);
        PL_Ku = sqrt(10.^(Path_loss(distance_Ku) / 10));
        for k = 1:Ku
            H_comm(:, :, k) = H_comm(:, :, k) * PL_Ku(k);
        end

        % ===== Baseline: perfect clutter kernel =====
        base_perf = baseline_comm_heur(para, H_comm, Vcc, R_eta, tarPara, a_t, b_t);
        local_scnr_comm = base_perf.SCNR_comm;
        local_scnr_heur_perf = base_perf.SCNR_heur;

        % ===== Baseline: estimated clutter kernel =====
        base_esti = baseline_comm_heur(para, H_comm, Vcc_hat, R_eta, tarPara, a_t, b_t);
        local_scnr_heur_esti = compute_SCNR(base_esti.RX_heur, base_esti.u_heur, ...
            Vcc, R_eta, sigma2_tar, a_t, b_t);

        % ===== ISAC: perfect clutter kernel =====
        outISAC_perf = solve_BLP_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, ...
            H_comm, gamma, sigma2_comm, opts);
        local_scnr_isac_perf = compute_SCNR(outISAC_perf.RX, outISAC_perf.u, ...
            Vcc, R_eta, sigma2_tar, a_t, b_t);

        % ===== ISAC: estimated clutter kernel =====
        outISAC_esti = solve_BLP_AO(para, sigma2_tar, Vcc_hat, R_eta, a_t, b_t, ...
            H_comm, gamma, sigma2_comm, opts);
        local_scnr_isac_esti = compute_SCNR(outISAC_esti.RX, outISAC_esti.u, ...
            Vcc, R_eta, sigma2_tar, a_t, b_t);

        % ===== Radar-only: perfect clutter kernel =====
        outRadar_perf = solve_Radar_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, opts);
        local_scnr_radar_perf = outRadar_perf.scnr_hist(end);

        % ===== Radar-only: estimated clutter kernel =====
        outRadar_esti = solve_Radar_AO(para, sigma2_tar, Vcc_hat, R_eta, a_t, b_t, opts);
        local_scnr_radar_esti = compute_SCNR(outRadar_esti.RX, outRadar_esti.u, ...
            Vcc, R_eta, sigma2_tar, a_t, b_t);

        scnr_comm_sum = scnr_comm_sum + local_scnr_comm;
        scnr_heur_perf_sum = scnr_heur_perf_sum + local_scnr_heur_perf;
        scnr_heur_esti_sum = scnr_heur_esti_sum + local_scnr_heur_esti;
        scnr_isac_perf_sum = scnr_isac_perf_sum + local_scnr_isac_perf;
        scnr_isac_esti_sum = scnr_isac_esti_sum + local_scnr_isac_esti;
        scnr_radar_perf_sum = scnr_radar_perf_sum + local_scnr_radar_perf;
        scnr_radar_esti_sum = scnr_radar_esti_sum + local_scnr_radar_esti;
    end

    % Average over Monte Carlo trials
    SCNR_ISAC_perf_ave(ii) = scnr_isac_perf_sum / ITER;
    SCNR_ISAC_esti_ave(ii) = scnr_isac_esti_sum / ITER;

    SCNR_radar_perf_ave(ii) = scnr_radar_perf_sum / ITER;
    SCNR_radar_esti_ave(ii) = scnr_radar_esti_sum / ITER;

    SCNR_heur_perf_ave(ii) = scnr_heur_perf_sum / ITER;
    SCNR_heur_esti_ave(ii) = scnr_heur_esti_sum / ITER;

    SCNR_comm_ave(ii) = scnr_comm_sum / ITER;
end

%% Plot SCNR versus transmit power
h_scnrPtx = figure('Color', 'white');
plot(Ptx_dBm_range, 10 * log10(SCNR_ISAC_perf_ave), '-o', 'Color', colors{2}, 'LineWidth', 1.5);
hold on;
plot(Ptx_dBm_range, 10 * log10(SCNR_ISAC_esti_ave), '--o', 'Color', colors{2}, 'LineWidth', 1.5);

plot(Ptx_dBm_range, 10 * log10(SCNR_radar_perf_ave), '-s', 'Color', colors{1}, 'LineWidth', 1.5);
plot(Ptx_dBm_range, 10 * log10(SCNR_radar_esti_ave), '--s', 'Color', colors{1}, 'LineWidth', 1.5);

plot(Ptx_dBm_range, 10 * log10(SCNR_heur_perf_ave), '-d', 'Color', colors{3}, 'LineWidth', 1.5);
plot(Ptx_dBm_range, 10 * log10(SCNR_heur_esti_ave), '--d', 'Color', colors{3}, 'LineWidth', 1.5);

plot(Ptx_dBm_range, 10 * log10(SCNR_comm_ave), '-^', 'Color', colors{4}, 'LineWidth', 1.5);

grid on;
xlabel('Transmit power (dBm)', 'FontSize', 12, 'FontName', 'Times New Roman');
ylabel('SCNR (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
legend('ISAC (perfect)', 'ISAC (estimated)', ...
       'Radar-only (perfect)', 'Radar-only (estimated)', ...
       'Heuristic (perfect)', 'Heuristic (estimated)', 'Comm-only', ...
       'Location', 'best', 'FontSize', 12, 'FontName', 'Times New Roman');