clc; clear; close all;

%% =======================
% Add project folders to MATLAB path  <<<【新增】
%% =======================
currentFilePath = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(currentFilePath, '..')));

%% =======================
% Common Parameters
%% =======================
SNR_dB = [-10 -5 0 5 10 15 20 25 30];

% ---- Comparison point ----
gamma_cmp = 0.8;   % ISAC-OFDM sensing weight
alpha_cmp = 0.8;   % ISAC-FDM comm bandwidth ratio

%% =======================
% Run ISAC-OFDM (waveform)
%% =======================
fprintf('Running ISAC-OFDM (gamma = %.1f)\n', gamma_cmp);
[BER_OFDM] = run_ISAC_OFDM(gamma_cmp, SNR_dB);

%% =======================
% Run ISAC-FDM (waveform)
%% =======================
fprintf('Running ISAC-FDM (alpha = %.1f)\n', alpha_cmp);
[BER_FDM] = run_ISAC_FDM(alpha_cmp, SNR_dB);

%% =======================
% Plot Comparison
%% =======================
figure;
semilogy(SNR_dB, BER_OFDM, '-o', 'LineWidth',1.8); hold on;
semilogy(SNR_dB, BER_FDM , '-s', 'LineWidth',1.8);
grid on;

xlabel('SNR (dB)');
ylabel('BER');
legend( ...
    sprintf('ISAC-OFDM (\\gamma = %.1f)', gamma_cmp), ...
    sprintf('ISAC-FDM (\\alpha = %.1f)', alpha_cmp), ...
    'Location','southwest');

title('ISAC-OFDM vs ISAC-FDM (Waveform-level Comparison)');
