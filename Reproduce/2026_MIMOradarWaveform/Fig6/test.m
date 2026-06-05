clc;
clear;
close all;
addpath('./functions_2018ICC');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');

rng(1);

%% 1. Parameter Settings
Kc = 4;
M = 12;
L = 40;
Pt = 1;

Q = 4;
rho = 0.2;
par = 1.1;

c = ones(M, 1) * Pt / M;

d = 0.5;
lambda = 2 * d;
pos = (0:M - 1) * d;
normalizedPos = pos / lambda;

afun = @(theta) exp(1j * pi * (0:M - 1)' * sind(theta));

%% Desired Beampattern
theta_est = [-60, 0, 60];
Kt = length(theta_est);

Delta = 5;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));

idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - Delta & theta_grid <= theta_est(i) + Delta;
end
P_des(idx) = 1;

%% Omni-Directional Beampattern
OmniRd = (Pt / M) * eye(M);

%% Directional Beampattern
w_l = ones(length(theta_grid), 1);
w_c = 0;
[DirectRd1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);
P_des1 = P_des * alpha1;

DirectRd2 = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2 / M;
DirectRd2 = (DirectRd2 + DirectRd2') / 2;
DirectRd2 = projectToPSD(DirectRd2);
DirectRd2 = DirectRd2 + 1e-10 * eye(M);
DirectRd2 = Pt * DirectRd2 / real(trace(DirectRd2));

[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt);
fprintf('trace(DirectRd3) = %.6f\n', trace(DirectRd3));

%% Choose Directional Covariance Matrix
DirectRd = DirectRd2;

%% SNR Settings
SNRdB = -5:1:12;
N0 = Pt ./ 10.^(SNRdB / 10);

%% Initialization


Iters = 100;

OmniStrictSERArray = zeros(Iters, length(SNRdB));
OmniTradeoffSERTolArray = zeros(Iters, length(SNRdB));
OmniTradeoffSERPerAntArray = zeros(Iters, length(SNRdB));
DirectStrictSERArray = zeros(Iters, length(SNRdB));
DirectTradeoffSERTolArray = zeros(Iters, length(SNRdB));
DirectTradeoffSERPerAntArray = zeros(Iters, length(SNRdB));
ZeroMUISERArray = zeros(Iters, length(SNRdB));

OmniStrictBPArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPTolArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));
DirectStrictBPArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPTolArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));

%% Monte Carlo Simulation
for iter = 1:Iters
    fprintf('Monte Carlo iteration: %d / %d\n', iter, Iters);

    H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);

    data = randi([0, Q - 1], Kc, L);
    S = pskmod(data, Q, pi / Q, 'gray');

    OmniStrictX = strict_waveform(H, S, OmniRd, L);
    DirectStrictX = strict_waveform(H, S, DirectRd, L);

    % 如果你想用波形合成生成严格雷达波形，可以替换为下面两行
    % OmniStrictX = WaveformSynthesisXoptimR(OmniRd, L, par);
    % DirectStrictX = WaveformSynthesisXoptimR(DirectRd, L, par);

    OmniTradeoffTolX = algorithm1_tradeoff(H, S, OmniStrictX, Pt, rho);
    DirectTradeoffTolX = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);

    OmniTradeoffPerAntX = helperRadComWaveform(H, S, OmniStrictX, Pt, rho);
    DirectTradeoffPerAntX = helperRadComWaveform(H, S, DirectStrictX, Pt, rho);

    for idxSNR = 1:length(SNRdB)
        OmniStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniStrictX, data, Q, N0(idxSNR));
        OmniTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffTolX, data, Q, N0(idxSNR));
        OmniTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffPerAntX, data, Q, N0(idxSNR));
        DirectStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectStrictX, data, Q, N0(idxSNR));
        DirectTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffTolX, data, Q, N0(idxSNR));
        DirectTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffPerAntX, data, Q, N0(idxSNR));
        ZeroMUISERArray(iter, idxSNR) = qpsk_ser_zero_mui(S, data, Q, N0(idxSNR));
    end

    OmniStrictR = OmniStrictX * OmniStrictX' / L;
    OmniTradeoffTolR = OmniTradeoffTolX * OmniTradeoffTolX' / L;
    OmniTradeoffPerAntR = OmniTradeoffPerAntX * OmniTradeoffPerAntX' / L;
    DirectStrictR = DirectStrictX * DirectStrictX' / L;
    DirectTradeoffTolR = DirectTradeoffTolX * DirectTradeoffTolX' / L;
    DirectTradeoffPerAntR = DirectTradeoffPerAntX * DirectTradeoffPerAntX' / L;

    OmniStrictBPArray(iter, :) = beampattern_linear(OmniStrictR, afun, theta_grid);
    OmniTradeoffBPTolArray(iter, :) = beampattern_linear(OmniTradeoffTolR, afun, theta_grid);
    OmniTradeoffBPPerAntArray(iter, :) = beampattern_linear(OmniTradeoffPerAntR, afun, theta_grid);
    DirectStrictBPArray(iter, :) = beampattern_linear(DirectStrictR, afun, theta_grid);
    DirectTradeoffBPTolArray(iter, :) = beampattern_linear(DirectTradeoffTolR, afun, theta_grid);
    DirectTradeoffBPPerAntArray(iter, :) = beampattern_linear(DirectTradeoffPerAntR, afun, theta_grid);
end

%% Average Results
OmniStrictSER = mean(OmniStrictSERArray, 1);
OmniTradeoffSERTol = mean(OmniTradeoffSERTolArray, 1);
OmniTradeoffSERPerAnt = mean(OmniTradeoffSERPerAntArray, 1);
DirectStrictSER = mean(DirectStrictSERArray, 1);
DirectTradeoffSERTol = mean(DirectTradeoffSERTolArray, 1);
DirectTradeoffSERPerAnt = mean(DirectTradeoffSERPerAntArray, 1);
ZeroMUISER = mean(ZeroMUISERArray, 1);

%% Figure: SER
figure(1);
semilogy(SNRdB, OmniStrictSER, 'b-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectStrictSER, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, OmniTradeoffSERTol, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERTol, 'r-d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, OmniTradeoffSERPerAnt, 'b--s', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERPerAnt, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, ZeroMUISER, 'k--v', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Transmit SNR (dB)');
ylabel('SER');
legend('Omni-Strict', ...
       'Directional-Strict', ...
       'Omni-Tradeoff-Total, \rho = 0.2', ...
       'Directional-Tradeoff-Total, \rho = 0.2', ...
       'Omni-Tradeoff-perAnt, \rho = 0.2', ...
       'Directional-Tradeoff-perAnt, \rho = 0.2', ...
       'Zero MUI', ...
       'Location', 'SouthWest');
xlim([min(SNRdB), max(SNRdB)]);
ylim([1e-5, 1]);

%% Figure: Beampattern
OmniStrictBP = mean(OmniStrictBPArray, 1);
OmniTradeoffBPTol = mean(OmniTradeoffBPTolArray, 1);
OmniTradeoffBPPerAnt = mean(OmniTradeoffBPPerAntArray, 1);
DirectStrictBP = mean(DirectStrictBPArray, 1);
DirectTradeoffBPTol = mean(DirectTradeoffBPTolArray, 1);
DirectTradeoffBPPerAnt = mean(DirectTradeoffBPPerAntArray, 1);

figure(2);
plot(theta_grid, 10 * log10(OmniStrictBP + eps), 'b-', 'LineWidth', 1.5);
hold on;
plot(theta_grid, 10 * log10(DirectStrictBP + eps), 'r-', 'LineWidth', 1.5);
plot(theta_grid, 10 * log10(OmniTradeoffBPTol + eps), 'b--', 'LineWidth', 1.5);
plot(theta_grid, 10 * log10(DirectTradeoffBPTol + eps), 'r--', 'LineWidth', 1.5);
plot(theta_grid, 10 * log10(OmniTradeoffBPPerAnt + eps), 'c--', 'LineWidth', 1.5);
plot(theta_grid, 10 * log10(DirectTradeoffBPPerAnt + eps), 'm--', 'LineWidth', 1.5);
grid on;
xlabel('\theta (deg)');
ylabel('Beampattern (dB)');
legend('Omni-Strict', ...
       'Directional-Strict', ...
       'Omni-Tradeoff-Total, \rho = 0.2', ...
       'Directional-Tradeoff-Total, \rho = 0.2', ...
       'Omni-Tradeoff-perAnt, \rho = 0.2', ...
       'Directional-Tradeoff-perAnt, \rho = 0.2', ...
       'Location', 'South');
xlim([-90, 90]);


function BP = beampattern_linear(R, afun, theta_grid)
    BP = zeros(size(theta_grid));
    for idxTheta = 1:length(theta_grid)
        a = afun(theta_grid(idxTheta));
        BP(idxTheta) = real(a' * R * a);
    end
end