


clc;
clear;
close all;
addpath('./functions');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');

%% 1. 参数设置（示例，可修改）
Kc = 6;                      % # of users
M = 12;                     % 天线数
L = 40;                     % # of Communication Frame
Pt  = 1;
c = ones(M, 1) * Pt/M;        % 对角元固定值
% c = rand(M, 1)
Iters =100;

d = 0.5;
lambda = 2 * d;
pos = (0:M-1) * d;
normalizedPos = pos / lambda;

afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1
rho = 0.1;   % Tradeoff Settings

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

%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  helperMMSECovariance 默认 diag(R)=1, trace(R)=M
%  为了和文献1对齐，将 R 除以 M，使 trace(R)=1
DirectRd2_raw = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2_raw / M;
DirectRd2 = (DirectRd2 + DirectRd2')/2;

%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  不用 cos(theta) 权重，不做积分归一化，不用 barrier/Newton，直接 CVX 最小化二范数
[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt); 
fprintf('trace(Rmmse1) = %.6f\n',  trace(DirectRd3));

if 1
    %% 计算 beampattern
    P_lit1 = zeros(size(theta_grid));
    P_lit2 = zeros(size(theta_grid));
    P_lit2_ = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        ai = afun(theta_grid(i));
        P_lit1(i) = real(ai' * DirectRd1 * ai);
        P_lit2(i) = real(ai' * DirectRd2 * ai);
        P_lit2_(i) = real(ai' * DirectRd3 * ai);
    end
    % P_lit1(P_lit1 < 0) = 0;
    % P_lit2(P_lit2 < 0) = 0;
    % P_lit2_(P_lit2 < 0) = 0;

    P_des1 = alpha1 * P_des;
    figure(1);
    plot(theta_grid, P_des1, 'k--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_lit1, 'b-', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_lit2, 'r-.', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_lit2_, 'c-.', 'LineWidth', 1.5);
    grid on;
    xlabel('\theta (degrees)');
    ylabel('Beampattern');
    legend('Desired', 'Beampattern Matching', 'Squared Error', 'my Squared Error', 'Location', 'best');
    title('Comparison under trace(R)=1');
    xlim([-90, 90]);
end

SNRdB = -5:1:12;
N0 = Pt ./ 10.^(SNRdB/10);

OmniStrictCapacityArray = zeros(Iters, length(SNRdB));
OmniTradeoffCapacityArray = zeros(Iters, length(SNRdB));
DirectStrictCapacityArray = zeros(Iters, length(SNRdB));
DirectTradeoffCapacityArray = zeros(Iters, length(SNRdB));

OmniStrictBPArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPArray = zeros(Iters, length(theta_grid));
DirectStrictBPArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPArray = zeros(Iters, length(theta_grid));


%% Choose Directional Covariance Matrix
DirectRd = DirectRd2;

%% Monte Carlo Simulation
for iter = 1:Iters
    fprintf('Monte Carlo iteration: %d / %d\n', iter, Iters);
    H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);
    data = randi([0, 3], Kc, L);
    S = pskmod(data, 4, pi / 4, 'gray');
    OmniStrictX = strict_waveform(H, S, OmniRd, L);
    DirectStrictX = strict_waveform(H, S, DirectRd, L);
    OmniTradeoffX = algorithm1_tradeoff(H, S, OmniStrictX, Pt, rho);
    DirectTradeoffX = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);
    for idxSNR = 1:length(SNRdB)
        OmniStrictCapacityArray(iter, idxSNR) = average_user_rate(H, OmniStrictX, S, N0(idxSNR));
        OmniTradeoffCapacityArray(iter, idxSNR) = average_user_rate(H, OmniTradeoffX, S, N0(idxSNR));
        DirectStrictCapacityArray(iter, idxSNR) = average_user_rate(H, DirectStrictX, S, N0(idxSNR));
        DirectTradeoffCapacityArray(iter, idxSNR) = average_user_rate(H, DirectTradeoffX, S, N0(idxSNR));
    end
    OmniStrictR = OmniStrictX * OmniStrictX' / L;
    OmniTradeoffR = OmniTradeoffX * OmniTradeoffX' / L;
    DirectStrictR = DirectStrictX * DirectStrictX' / L;
    DirectTradeoffR = DirectTradeoffX * DirectTradeoffX' / L;
    OmniStrictBPArray(iter, :) = beampattern_dB(OmniStrictR, afun, theta_grid);
    OmniTradeoffBPArray(iter, :) = beampattern_dB(OmniTradeoffR, afun, theta_grid);
    DirectStrictBPArray(iter, :) = beampattern_dB(DirectStrictR, afun, theta_grid);
    DirectTradeoffBPArray(iter, :) = beampattern_dB(DirectTradeoffR, afun, theta_grid);
end

%% Average Results
OmniStrictCapacity = mean(OmniStrictCapacityArray, 1);
OmniTradeoffCapacity = mean(OmniTradeoffCapacityArray, 1);
DirectStrictCapacity = mean(DirectStrictCapacityArray, 1);
DirectTradeoffCapacity = mean(DirectTradeoffCapacityArray, 1);
OmniStrictBP = mean(OmniStrictBPArray, 1);
OmniTradeoffBP = mean(OmniTradeoffBPArray, 1);
DirectStrictBP = mean(DirectStrictBPArray, 1);
DirectTradeoffBP = mean(DirectTradeoffBPArray, 1);
AWGNCapacity = log2(1 + Pt ./ N0);

%% Figure 2: Average Achievable Rate
figure(2);
plot(SNRdB, AWGNCapacity, 'r--v', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, OmniTradeoffCapacity, 'k--x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, DirectTradeoffCapacity, 'b--o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, OmniStrictCapacity, 'k-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNRdB, DirectStrictCapacity, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7);
grid on;
xlabel('Transmit SNR (dB)');
ylabel('Average Achievable Rate (bps/Hz/user)');
legend('AWGN Capacity', 'Omni-Tradeoff, \rho = 0.1', 'Directional-Tradeoff, \rho = 0.1', 'Omni-Strict', 'Directional-Strict', 'Location', 'NorthWest');
xlim([min(SNRdB), max(SNRdB)]);

%% Figure 3: Radar Beampattern
figure(3);
plot(theta_grid, P_des1, 'k-', 'LineWidth', 1.5); hold on;
plot(theta_grid, OmniStrictBP, 'r-', 'LineWidth', 1.5); hold on;
plot(theta_grid, DirectStrictBP, 'g--', 'LineWidth', 1.5); hold on;
plot(theta_grid, OmniTradeoffBP, 'b--', 'LineWidth', 1.5); hold on;
plot(theta_grid, DirectTradeoffBP, 'k--', 'LineWidth', 1.5);
grid on;
xlabel('\theta (deg)');
ylabel('Beampattern');
legend('Desired', 'Omni-Strict', 'Directional-Strict', 'Omni-Tradeoff, \rho = 0.1', 'Directional-Tradeoff, \rho = 0.1', 'Location', 'South');
xlim([-90, 90]);

























