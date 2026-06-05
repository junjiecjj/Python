


clc;
clear;
close all;
addpath('./functions_2018ICC');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');


%% 1. 参数设置（示例，可修改）
Kc = 4;                      % # of users
M = 12;                     % 天线数
L = 40;                     % # of Communication Frame
Pt  = 1;
c = ones(M, 1) * Pt/M;        % 对角元固定值
% c = rand(M, 1)

d = 0.5;
lambda = 2 * d;
pos = (0:M-1) * d;
normalizedPos = pos / lambda;

afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

%% Desired Beampattern
theta_est = [-60, 0, 60];   % 目标角度估计（度）
Kt = length(theta_est);      % 目标个数

Delta = 5;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i)-Delta & theta_grid <= theta_est(i)+Delta;
end
P_des(idx) = 1;

%% Omni-Directional Beampattern
OmniRd = (Pt / M) * eye(M);

%% Directional Beampattern
%  文献1：On Probing Signal Design For MIMO Radar, C. Beampattern Matching Design
%  diag(R)=1/M, trace(R)=1, wc=0
w_l = ones(length(theta_grid), 1);
w_c = 0;
[DirectRd1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);
P_des1 = P_des * alpha1;

%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  helperMMSECovariance 默认 diag(R)=1, trace(R)=M,为了和文献1对齐，将 R 除以 M，使 trace(R)=1
DirectRd2 = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2 / M;
DirectRd2 = projectToPSD(DirectRd2);
DirectRd2 = DirectRd2 + 1e-10 * eye(size(M));
%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  不用 cos(theta) 权重，不做积分归一化，不用 barrier/Newton，直接 CVX 最小化二范数
[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt); 
fprintf('trace(Rmmse1) = %.6f\n',  trace(DirectRd3));

SNRdB = -5:1:18;
N0 = Pt ./ 10.^(SNRdB/10);


%% Choose Directional Covariance Matrix
DirectRd = DirectRd2;
rho = 0.2;   % Tradeoff Settings
par = 1.1;                          % Parameter that controls low PAR

%% Monte Carlo Simulation

%% Initialization

Q = 4;
Iters = 1000;

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
    for idxSNR = 1:length(SNRdB)
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

        OmniStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniStrictX, data, Q, N0(idxSNR));
        OmniTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffTolX, data, Q, N0(idxSNR));
        OmniTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffPerAntX, data, Q, N0(idxSNR));
        DirectStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectStrictX, data, Q, N0(idxSNR));
        DirectTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffTolX, data, Q, N0(idxSNR));
        DirectTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffPerAntX, data, Q, N0(idxSNR));
        ZeroMUISERArray(iter, idxSNR) = qpsk_ser_zero_mui(S, data, Q, N0(idxSNR));
    end

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
semilogy(SNRdB, OmniTradeoffSERTol, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, OmniTradeoffSERPerAnt, 'b--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectStrictSER, 'r-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERTol, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERPerAnt, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, ZeroMUISER, 'k--v', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Transmit SNR (dB)');
ylabel('SER');
legend('Omni-Strict', ...
       'Omni-Tradeoff-Total, \rho = 0.2', ...
       'Omni-Tradeoff-perAnt, \rho = 0.2', ...
       'Directional-Strict', ...
       'Directional-Tradeoff-Total, \rho = 0.2', ...
       'Directional-Tradeoff-perAnt, \rho = 0.2', ...
       'Zero MUI', ...
       'Location', 'SouthWest');
xlim([min(SNRdB), max(SNRdB)]);
ylim([1e-5, 1]);












