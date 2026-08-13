clear
close all
clc

% Parameters
B = 20e6;           % Bandwidth (Hz)
T = 1e-6;           % Pulse duration (s)
TB = B * T;         % Time-bandwidth product
fs = 2 * B;         % Sampling rate (Hz)
t = 0:1/fs:T-1/fs;  % Time vector

% FMCW Signal
f0 = 0;             % Start frequency (Hz)
f1 = B;             % End frequency (Hz)
fmcw_signal = chirp(t, f0, T, f1, 'linear','complex');

% PMCW Signal
pmcw_signal = exp(1j * 2 * pi * rand(1, length(t))); % Random phase modulation

% Ambiguity Function for FMCW
[af_fmcw, delay_fmcw, doppler_fmcw] = ambgfun(fmcw_signal, fs, 1/T);

% Ambiguity Function for PMCW
[af_pmcw, delay_pmcw, doppler_pmcw] = ambgfun(pmcw_signal, fs, 1/T);

% Plot Ambiguity Function for FMCW
figure;
imagesc(delay_fmcw, doppler_fmcw, abs(af_fmcw));
axis xy;
xlabel('Normalized Delay', 'Interpreter', 'Latex');
ylabel('Normalized Frequency', 'Interpreter', 'Latex');
xticks([min(delay_fmcw), 0, max(delay_fmcw)]);
xticklabels({'-\tau/2', '0', '\tau/2'});
yticks([min(doppler_fmcw), 0, max(doppler_fmcw)]);
yticklabels({'-B/2', '0', 'B/2'});
colorbar;

% Plot Ambiguity Function for PMCW
figure;
imagesc(delay_pmcw, doppler_pmcw, abs(af_pmcw));
axis xy;
xlabel('Normalized Delay', 'Interpreter', 'Latex');
ylabel('Normalized Frequency', 'Interpreter', 'Latex');
xticks([min(delay_pmcw), 0, max(delay_pmcw)]);
xticklabels({'-\tau/2', '0', '\tau/2'});
yticks([min(doppler_pmcw), 0, max(doppler_pmcw)]);
yticklabels({'-B/2', '0', 'B/2'});
colorbar;