


clc;
clear;
close all;

rng(42);

addpath('./functions_2018ICC');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');


%% 1. 参数设置（示例，可修改）
Kc = 6;                      % # of users
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

SNRdB = -5:1:12;
N0 = Pt ./ 10.^(SNRdB/10);
Iters =100;

OmniStrictCapacityArray = zeros(Iters, length(SNRdB));
OmniTradeoffCapacityTolArray = zeros(Iters, length(SNRdB));
OmniTradeoffCapacityPerAntArray = zeros(Iters, length(SNRdB));
DirectStrictCapacityArray = zeros(Iters, length(SNRdB));
DirectTradeoffCapacityTolArray = zeros(Iters, length(SNRdB));
DirectTradeoffCapacityPerAntArray = zeros(Iters, length(SNRdB));

OmniStrictBPArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPTolArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));
DirectStrictBPArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPTolArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));

%% Choose Directional Covariance Matrix
DirectRd = DirectRd2;
rho = 0.2;   % Tradeoff Settings
par = 1.1;                          % Parameter that controls low PAR

%% Monte Carlo Simulation
for iter = 1:Iters
    fprintf('Monte Carlo iteration: %d / %d\n', iter, Iters);
    H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);
    data = randi([0, 3], Kc, L);
    S = pskmod(data, 4, pi / 4, 'gray');
    
    % 生成严格满足雷达约束R但是尽可能小的MUI的波形；
    OmniStrictX = strict_waveform(H, S, OmniRd, L);
    DirectStrictX = strict_waveform(H, S, DirectRd, L);

    % OmniStrictX = helperCAWaveformSynthesis(OmniRd, L, par);
    % DirectStrictX = helperCAWaveformSynthesis(DirectRd, L, par);

    % OmniStrictX = WaveformSynthesisXoptimR(OmniRd, L,  par );
    % DirectStrictX = WaveformSynthesisXoptimR(DirectRd, L, par);

    % 根据严格波形生成折中波形；
    OmniTradeoffTolX = algorithm1_tradeoff(H, S, OmniStrictX, Pt, rho);
    DirectTradeoffTolX = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);
    
    OmniTradeoffPerAntX = helperRadComWaveform(H, S, OmniStrictX, Pt, rho);
    DirectTradeoffPerAntX = helperRadComWaveform(H, S, DirectStrictX, Pt, rho);


    for idxSNR = 1:length(SNRdB)
        OmniStrictCapacityArray(iter, idxSNR) = average_user_rate(H, OmniStrictX, S, N0(idxSNR));
        OmniTradeoffCapacityTolArray(iter, idxSNR) = average_user_rate(H, OmniTradeoffTolX, S, N0(idxSNR));
        OmniTradeoffCapacityPerAntArray(iter, idxSNR) = average_user_rate(H, OmniTradeoffPerAntX, S, N0(idxSNR));
        DirectStrictCapacityArray(iter, idxSNR) = average_user_rate(H, DirectStrictX, S, N0(idxSNR));
        DirectTradeoffCapacityTolArray(iter, idxSNR) = average_user_rate(H, DirectTradeoffTolX, S, N0(idxSNR));
        DirectTradeoffCapacityPerAntArray(iter, idxSNR) = average_user_rate(H, DirectTradeoffPerAntX, S, N0(idxSNR));
    end
    OmniStrictR = OmniStrictX * OmniStrictX' / L;
    OmniTradeoffTolR = OmniTradeoffTolX * OmniTradeoffTolX' / L;
    OmniTradeoffPerAntR = OmniTradeoffPerAntX * OmniTradeoffPerAntX' / L;
    DirectStrictR = DirectStrictX * DirectStrictX' / L;
    DirectTradeoffTolR = DirectTradeoffTolX * DirectTradeoffTolX' / L;
    DirectTradeoffPerAntR = DirectTradeoffPerAntX * DirectTradeoffPerAntX' / L;

    OmniStrictBPArray(iter, :) = beampattern_dB(OmniStrictR, afun, theta_grid);
    OmniTradeoffBPTolArray(iter, :) = beampattern_dB(OmniTradeoffTolR, afun, theta_grid);
    OmniTradeoffBPPerAntArray(iter, :) = beampattern_dB(OmniTradeoffPerAntR, afun, theta_grid);
    DirectStrictBPArray(iter, :) = beampattern_dB(DirectStrictR, afun, theta_grid);
    DirectTradeoffBPTolArray(iter, :) = beampattern_dB(DirectTradeoffTolR, afun, theta_grid);
    DirectTradeoffBPPerAntArray(iter, :) = beampattern_dB(DirectTradeoffPerAntR, afun, theta_grid);

end

%% Average Results
OmniStrictCapacity = mean(OmniStrictCapacityArray, 1);
OmniTradeoffCapacityTol = mean(OmniTradeoffCapacityTolArray, 1);
OmniTradeoffCapacityPerAnt = mean(OmniTradeoffCapacityPerAntArray, 1);
DirectStrictCapacity = mean(DirectStrictCapacityArray, 1);
DirectTradeoffCapacityTol = mean(DirectTradeoffCapacityTolArray, 1);
DirectTradeoffCapacityPerAnt = mean(DirectTradeoffCapacityPerAntArray, 1);

OmniStrictBP = mean(OmniStrictBPArray, 1);
OmniTradeoffBPTol = mean(OmniTradeoffBPTolArray, 1);
OmniTradeoffBPPerAnt = mean(OmniTradeoffBPPerAntArray, 1);
DirectStrictBP = mean(DirectStrictBPArray, 1);
DirectTradeoffBPTol = mean(DirectTradeoffBPTolArray, 1);
DirectTradeoffBPPerAnt = mean(DirectTradeoffBPPerAntArray, 1);

AWGNCapacity = log2(1 + Pt ./ N0);

%% Figure 2: Average Achievable Rate
figure(1);
plot(SNRdB, AWGNCapacity, 'k--', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, OmniStrictCapacity, 'b--x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, OmniTradeoffCapacityTol, 'b--o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, OmniTradeoffCapacityPerAnt, 'b--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
 
plot(SNRdB, DirectStrictCapacity, 'r-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, DirectTradeoffCapacityTol, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, DirectTradeoffCapacityPerAnt, 'r-d', 'LineWidth', 1.5, 'MarkerSize', 7);

grid on;
xlabel('Transmit SNR (dB)');
ylabel('Average Achievable Rate (bps/Hz/user)');
legend('AWGN Capacity', 'Omni-Strict', 'Omni-Tradeoff-Tol, \rho = 0.2', 'Omni-Tradeoff-Per, \rho = 0.2', 'Directional-Strict', 'Directional-Tradeoff-Tol, \rho = 0.2', 'Directional-Tradeoff-Per, \rho = 0.2', 'Location', 'NorthWest');
xlim([min(SNRdB), max(SNRdB)]);

%% Figure 3: Radar Beampattern
figure(2);
plot(theta_grid, 10 * log10(P_des1 + eps), 'k-', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(OmniStrictBP + eps), 'b--', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(OmniTradeoffBPTol + eps), 'r-.', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(OmniTradeoffBPPerAnt + eps), 'g--', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(DirectStrictBP + eps), 'c-', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(DirectTradeoffBPTol + eps), 'm-', 'LineWidth', 1.5); hold on;
plot(theta_grid, 10 * log10(DirectTradeoffBPPerAnt + eps), 'y-', 'LineWidth', 1.5);
grid on;
xlabel('\theta (deg)');
ylabel('Beampattern');
legend('Desired', 'Omni-Strict', 'Omni-Tradeoff-Tol, \rho = 0.2', 'Omni-Tradeoff-Per, \rho = 0.2', 'Directional-Strict', 'Directional-Tradeoff-Tol, \rho = 0.2', 'Directional-Tradeoff-Per, \rho = 0.2', 'Location', 'Best');
xlim([-90, 90]);
ylim([-30, 10]);





















