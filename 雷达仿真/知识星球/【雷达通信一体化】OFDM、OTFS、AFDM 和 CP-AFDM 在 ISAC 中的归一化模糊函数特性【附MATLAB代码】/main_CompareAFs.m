% =========================================================================
% File Description: Main script to run and generate the smooth AFs
% Author    : Dr. (Eric) Hyeon Seok Rou
% Version   : v1.0
% Date      : Oct 13, 2025
% =========================================================================
%
close all; clear; clc;
% rehash toolboxcache
addpath "./helperFunctions/" ;

%% System Parametrisation
N  = 144;                   % number of discrete time samples
OS_dop = 4;                 % smoothing factor in doppler domain
OS_del = 4;                 % smoothing factor in delay domain
allones = true;             % If "true", transmit all one symbols, if "false", transmit random QAM symbols
M = 16;                     % Modulation order for 16-QAM
%
params = struct();
params.Ts    = 1e-6;           % arbitrary sampling period (For physical translation)
params.N     = N;           % number of time samples used in the AF (length(s))
params.Fs    = 1/params.Ts; % sampling rate [Hz]

%% Generate Signals
if allones
    symbols = ones(N, 1);
else
    symbols = randi([0 M-1], N, 1); 
    symbols = qammod(symbols, M);
end

%---- OFDM
IDFT_matrix = dftmtx(N)'/sqrt(N);
s_OFDM = IDFT_matrix * symbols;

%---- OTFS
L_OTFS = sqrt(N);
FH_OTFS = kron( dftmtx(L_OTFS)'/sqrt(L_OTFS), eye(L_OTFS));
s_OTFS = FH_OTFS * symbols;

%---- AFDM
n  = (0:N-1).';
ellmax = 6;
fmax   = 4;
AFDM_guard = 1;
AFDM_resources = (2*(fmax + AFDM_guard)*ellmax) + (2*(fmax + AFDM_guard)) + (ellmax);
if AFDM_resources > N
    warning("AFDM orthogonality not satisfied (resources=%d > N=%d)", AFDM_resources, N);
end
c1 = (2*(fmax + AFDM_guard) + 1)/(2*N);
c2 = 1/(2*N);
chirp_c1 = exp(-2j*pi*c1*(n.^2));
chirp_c2 = exp(-2j*pi*c2*(n.^2));
s_AFDM = diag(chirp_c1)' * IDFT_matrix * diag(chirp_c2)' * symbols;

%---- CP-AFDM
c2_perm = randperm(N);
chirpc2_perm = chirp_c2(c2_perm);
s_CPAFDM = diag(chirp_c1)' * IDFT_matrix * diag(chirpc2_perm)' * symbols;

%% Obtain AFs and extract metrics using the helper functions
[tau_norm, nu_norm, OFDM_AFdB_zerodop, OFDM_AFdB_zerodel, OFDM_metrics_delay, OFDM_metrics_dopp] = AF_fullAnalysis(s_OFDM, params, OS_del, OS_dop);
[~, ~, AFDM_AFdB_zerodop, AFDM_AFdB_zerodel, AFDM_metrics_delay, AFDM_metrics_dopp] = AF_fullAnalysis(s_AFDM, params, OS_del, OS_dop);
[~, ~, CPAFDM_AFdB_zerodop, CPAFDM_AFdB_zerodel, CPAFDM_metrics_delay, CPAFDM_metrics_dopp] = AF_fullAnalysis(s_CPAFDM, params, OS_del, OS_dop);
[~, ~, OTFS_AFdB_zerodop, OTFS_AFdB_zerodel, OTFS_metrics_delay, OTFS_metrics_dopp] = AF_fullAnalysis(s_OTFS, params, OS_del, OS_dop);

%% ---------- Common Figure Settings ----------
fig_width  = 6;                   
fig_height = fig_width / 1.618;
lw = 1.4;
%
clr_OFDM   = [0, 0, 0];        % black
clr_OTFS   = [0, 0.35, 0.7];   % dark blue
clr_AFDM   = [0, 0.5, 0.1];    % dark green
clr_CPAFDM = [0.6, 0, 0];      % dark red
%
set(groot, 'defaultTextInterpreter','latex', 'defaultAxesTickLabelInterpreter','latex', 'defaultLegendInterpreter','latex', 'defaultAxesFontSize',10);

% Helpers
make_fig = @(name) figure('Color','white', 'Units','inches', 'Position',[1 1 fig_width fig_height], 'PaperUnits','inches', 'PaperSize',[fig_width fig_height], 'Name',name);

%% ---------- Plot: Delay AF (Zero-Doppler cut) ----------
h1 = make_fig('AFsingle_delay');
hold on; grid on; box on;
plot(tau_norm, OFDM_AFdB_zerodop,   'LineWidth', lw, 'Color', clr_OFDM,   'DisplayName','OFDM');
plot(tau_norm, OTFS_AFdB_zerodop,   'LineWidth', lw, 'Color', clr_OTFS,   'DisplayName','OTFS');
plot(tau_norm, AFDM_AFdB_zerodop,   'LineWidth', lw, 'Color', clr_AFDM,   'DisplayName','AFDM');
plot(tau_norm, CPAFDM_AFdB_zerodop, 'LineWidth', lw, 'Color', clr_CPAFDM, 'DisplayName','CP-AFDM');
xlabel('Normalized Delay'); ylabel('Magnitude [dB]');
xlim([-1, 1]); ylim([-60, 10]);
title('Zero-Doppler Cut (Delay Resolution)');
legend('show','Location','northeast');
set(gca,'Color','none','LineWidth',0.75);

%% ---------- Plot: Doppler AF (Zero-delay cut) ----------
h2 = make_fig('AFsingle_doppler');
hold on; grid on; box on;
plot(nu_norm, OFDM_AFdB_zerodel,   'LineWidth', lw, 'Color', clr_OFDM,   'DisplayName','OFDM');
plot(nu_norm, OTFS_AFdB_zerodel,   'LineWidth', lw, 'Color', clr_OTFS,   'DisplayName','OTFS');
plot(nu_norm, AFDM_AFdB_zerodel,   'LineWidth', lw, 'Color', clr_AFDM,   'DisplayName','AFDM');
plot(nu_norm, CPAFDM_AFdB_zerodel, 'LineWidth', lw, 'Color', clr_CPAFDM, 'DisplayName','CP-AFDM');
xlabel('Normalized Doppler'); ylabel('Magnitude [dB]');
xlim([-0.5, 0.5]); ylim([-60, 10]);
title('Zero-Delay Cut (Doppler Resolution)');
legend('show','Location','northeast');
set(gca,'Color','none','LineWidth',0.75);




