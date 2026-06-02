clc;
clear;
close all;
addpath('./functions');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
rng(42);

%% Figure 4: Trade-off of Omni-Directional Beampattern Design

%% 1. 参数设置（示例，可修改）
KcList = 6 : 2 : 10;         % # of users
M = 16;                     % 天线数
L = 100;                     % # of Communication Frame
Pt  = 1;
c = ones(M, 1) * Pt/M;       % 对角元固定值

% comm snr
SNRdB = 10;
N0 = Pt ./ 10.^(SNRdB/10);
% radar snr
RadarSNRdB = -20;
radarSNR = 10^(RadarSNRdB / 10);

Pfa = 1e-7;
thetaDetect = 0;

d = 0.5;
lambda = 2 * d;
pos = (0:M-1) * d;
normalizedPos = pos / lambda;
afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

%% Desired Beampattern
theta_est = [thetaDetect];   % 目标角度估计（度）
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

%% Directional Beampattern
% %  文献1：On Probing Signal Design For MIMO Radar, C. Beampattern Matching Design,  diag(R)=1/M, trace(R)=1, wc=0
w_l = ones(length(theta_grid), 1);
w_c = 0;
[DirectRd1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);
fprintf('trace(DirectRd1) = %.6f\n',  trace(DirectRd1));

% % 文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
% % % helperMMSECovariance 默认 diag(R)=1, trace(R)=M, 为了和文献1对齐，将 R 除以 M，使 trace(R)=1
DirectRd2_raw = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2_raw / M;
% DirectRd2 = (DirectRd2 + DirectRd2')/2;
fprintf('trace(DirectRd2) = %.6f\n',  trace(DirectRd2));

[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt); 
fprintf('trace(DirectRd3) = %.6f\n',  trace(DirectRd3));

DirectRd = DirectRd1;
%% Tradeoff Settings
rhoList = 0.1:0.02:0.9;

%% Simulation Settings
Iters = 2000;

OmniRateArray = zeros(Iters, length(rhoList), length(KcList));
OmniProbabilityArray = zeros(Iters, length(rhoList), length(KcList));

%% Monte Carlo Simulation
for iter = 1:Iters
    clc;
    disp(['Progress - ', num2str(iter), '/', num2str(Iters)]);
    for idxRho = 1:length(rhoList)
        rho = rhoList(idxRho);
        for idxKc = 1:length(KcList)
            Kc = KcList(idxKc);
            H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);
            data = randi([0, 3], Kc, L);
            S = pskmod(data, 4, pi / 4, 'gray');
            
            OmniStrictX = strict_waveform(H, S, DirectRd, L);
            OmniTradeoffX = algorithm1_tradeoff(H, S, OmniStrictX, Pt, rho);
            OmniRateArray(iter, idxRho, idxKc) = average_user_rate(H, OmniTradeoffX, S, N0);
            OmniProbabilityArray(iter, idxRho, idxKc) = radar_detection_probability_fig4(sqrt(M) * OmniTradeoffX, thetaDetect, radarSNR, Pfa);
        end
    end
end

%% Average Results
OmniRate = squeeze(mean(real(OmniRateArray), 1));
OmniProbability = squeeze(mean(real(OmniProbabilityArray), 1));

OmniRate1 = OmniRate(:, 1);
OmniRate2 = OmniRate(:, 2);
OmniRate3 = OmniRate(:, 3);
% OmniRate4 = OmniRate(:, 4);

OmniProbability1 = OmniProbability(:, 1);
OmniProbability2 = OmniProbability(:, 2);
OmniProbability3 = OmniProbability(:, 3);
% OmniProbability4 = OmniProbability(:, 4);

%% Figure 4
figure(1);
plot(OmniRate1, OmniProbability1, 'b', 'LineWidth', 1.5); hold on;
plot(OmniRate2, OmniProbability2, 'k', 'LineWidth', 1.5); hold on;
plot(OmniRate3, OmniProbability3, 'r', 'LineWidth', 1.5); hold on;
% plot(OmniRate4, OmniProbability4, 'g', 'LineWidth', 1.5); hold on;
grid on;
xlabel('Average Achievable Rate (bps/Hz/user)');
ylabel('Detection Probability');
legend('K=6', 'K=8', 'K=10', 'Location', 'southwest');

if 0
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

    P_des1 = alpha1 * P_des;
    figure(2);
    plot(theta_grid, 10 * log10(P_des1 + eps), 'k--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, 10 * log10(P_lit1 + eps), 'b-', 'LineWidth', 1.5); hold on;
    plot(theta_grid, 10 * log10(P_lit2 + eps), 'r-.', 'LineWidth', 1.5); hold on;
    plot(theta_grid, 10 * log10(P_lit2_ + eps), 'c-.', 'LineWidth', 1.5);
    grid on;
    xlabel('\theta (degrees)');
    ylabel('Beampattern');
    legend('Desired', 'Beampattern Matching', 'Squared Error', 'my Squared Error', 'Location', 'best');
    title('Comparison under trace(R)=1');
    xlim([-90, 90]);
    ylim([-40, 20]);
end





