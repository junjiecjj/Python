% This Matlab script is used to implement the BLP-based joint transmit and
% receive design for clutter-aware ISAC in the paper:
% R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-aware integrated sensing and communication: Models, methods, and future directions,” Proc. IEEE, to appear.
% Last edited by Peishi Li (lipeishi@mail.dlut.edu) in 2026-03-18

clear all;
clc;
close all;
rng('shuffle');

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'function'));
load(fullfile(root,'data','covMat_BLP.mat'), 'para', 'Vcc', 'Vcc_hat', 'R_eta')
%% Basic system parameters
N = 32;              % Number of subcarriers
L = 64;              % Number of OFDM symbols / slow-time snapshots
para.N = N;
para.L = L;
Nt = para.Nt;             % Number of transmit antennas
Nr = para.Nr;             % Number of receive antennas
fc = para.fc;        % Carrier frequency
deltaf = para.deltaf;% Subcarrier spacing
Ku = para.Ku;        % Number of communication users

% Communication settings
gamma_comm = 10^(10/10);   % Target SINR for communication users
gamma = gamma_comm * ones(N, Ku);
alpha_path = 2.7;          % Path-loss exponent
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
f_n = para.fc + (0:para.N-1) * para.deltaf;

% Target parameters
tarPara.M = 1;
tarPara.rcs = 10.^([10]/10);
tarPara.theta = [-10]';
tarPara.ranges = [41.8]';
tarPara.velos = [-31.2]';
tarPara.delays = 2 * tarPara.ranges / para.c0;
tarPara.dopplers = 2 * tarPara.velos / para.lambda_wave;
tarPara.alpha_mn = zeros(tarPara.M, para.N);
for m = 1:tarPara.M
    tarPara.alpha_mn(m, :) = sqrt(para.Pt * (para.c0./f_n).^2 *para.Gtx*para.Grx ...
        * tarPara.rcs(m) / ((4*pi)^3 * tarPara.ranges(m)^4));
end
sigma2_tar = abs(tarPara.alpha_mn).^2;

% Frequency-dependent steering vectors
chi_n = @(n) 1 + n*deltaf/fc;
a_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nt-1)'*chi_n(n) )./sqrt(Nt);
b_n = @(theta, n) exp(-1j * pi * sin(deg2rad(theta)) * (0:Nr-1)'*chi_n(n) )./sqrt(Nr);

% Generate communication channels
H_comm = (randn(Nt, N, Ku) + 1j*randn(Nt, N, Ku))/sqrt(2);
distance_Ku = randi([50, 400], 1, 1, Ku);
PL_Ku = Path_loss(distance_Ku);
PL_Ku = sqrt(10.^(PL_Ku/10));
for k = 1:Ku
    H_comm(:,:,k) = H_comm(:,:,k)*PL_Ku(k);
end

% Build target steering vectors over subcarriers
a_t = cell(N,1);
b_t = cell(N,1);
for n = 0:N-1
    a_t{n+1} = a_n(tarPara.theta, n);
    b_t{n+1} = b_n(tarPara.theta, n);
end
%% Baseline design
baseline = baseline_comm_heur(para, H_comm, Vcc, R_eta, tarPara, a_t, b_t);

fprintf('Perfect Vcc: SCNR comm-only = %.3f dB\n', 10*log10(baseline.SCNR_comm));
fprintf('Perfect Vcc: SCNR heuristic = %.3f dB\n', 10*log10(baseline.SCNR_heur));
%% BLP-based ISAC design with perfect clutter kernel
opts.verbose = 0;
outISAC = solve_BLP_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, H_comm, gamma, sigma2_comm, opts);

fprintf('\n===================== ISAC (perfect covMat) =====================\n');
fprintf('ISAC final SCNR with perfect covMat = %.3f dB\n\n', outISAC.scnr_hist_dB(end));
SINR_comm_ISAC = comp_SINR_comm(outISAC.RX, outISAC.Rnk_sdp, H_comm, sigma2_comm);
%% BLP-based ISAC design with estimated clutter kernel
opts_esti = opts;
opts_esti.verbose = 0;
outISAC_estiCov = solve_BLP_AO(para, sigma2_tar, Vcc_hat, R_eta, a_t, b_t, H_comm, gamma, sigma2_comm, opts);
SCNR_estiCov = compute_SCNR(outISAC_estiCov.RX, outISAC_estiCov.u, Vcc, R_eta, sigma2_tar, a_t, b_t);

fprintf('\n===================== ISAC (estimated covMat) =====================\n');
fprintf('ISAC final SCNR with estimated covMat = %.3f dB\n\n', 10*log10(SCNR_estiCov));
%% Radar-only design with perfect clutter kernel
optsR = opts;
optsR.verbose = 0;
outRadar = solve_Radar_AO(para, sigma2_tar, Vcc, R_eta, a_t, b_t, optsR);

fprintf('\n===================== Radar-only (perfect covMat) =====================\n');
fprintf('Radar-only final SCNR = %.3f dB\n\n', outRadar.scnr_hist_dB(end));

%% Radar-only design with estimated clutter kernel
outRadar_esti = solve_Radar_AO(para, sigma2_tar, Vcc_hat, R_eta, a_t, b_t, optsR);
SCNR_radar_estiCov = compute_SCNR(outRadar_esti.RX, outRadar_esti.u, Vcc, R_eta, sigma2_tar, a_t, b_t);

fprintf('\n===================== Radar-only (estimated covMat) =====================\n');
fprintf('Radar-only final SCNR = %.3f dB\n\n', 10*log10(SCNR_radar_estiCov));