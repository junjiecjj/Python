clc; clear; close all;

%% =========================
% Expected Trend Parameters
%% =========================
SNR_dB = -10:1:30;
SNR_lin = 10.^(SNR_dB/10);   % 注意：負 SNR 也完全 OK

gamma = 0.8;   % ISAC-OFDM sensing weight
alpha = 0.8;   % ISAC-FDM comm bandwidth ratio

Nbits = 64 * 300 * 2;   % approximate total bits (for lower bound)

%% =========================
% ISAC-FDM (frequency split)
%% =========================
% Effective SNR reduced by bandwidth loss
snr_eff_fdm = alpha * SNR_lin;

BER_FDM = qfunc(sqrt(2 * snr_eff_fdm));

% Finite-sample BER floor
BER_floor_fdm = 0.5 / (alpha * Nbits);
BER_FDM = max(BER_FDM, BER_floor_fdm);

%% =========================
% ISAC-OFDM (co-channel ISAC)
%% =========================
% Effective SNR penalty due to ISAC interference (gamma)
isac_penalty = (1 - gamma)^2;   % simple interference model
snr_eff_ofdm = isac_penalty * SNR_lin;

BER_OFDM = qfunc(sqrt(2 * snr_eff_ofdm));

% Finite-sample BER floor (full bandwidth)
BER_floor_ofdm = 0.5 / Nbits;
BER_OFDM = max(BER_OFDM, BER_floor_ofdm);

%% =========================
% Plot Expected Results
%% =========================
figure;
semilogy(SNR_dB, BER_OFDM, '-s', 'LineWidth',1.8); hold on;
semilogy(SNR_dB, BER_FDM,'-o', 'LineWidth',1.8);
grid on;

xlabel('SNR (dB)');
ylabel('BER');
legend( ...
    'ISAC-OFDM (Expected)', ...
    'ISAC-FDM (Expected)', ...
    'Location','southwest');

title('Expected BER Trends: ISAC-OFDM vs ISAC-FDM');
