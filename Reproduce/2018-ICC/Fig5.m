clc;
clear all;
close all;
addpath('./functions');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
rng(1);

%% Figure 5: Trade-off of Directional Beampattern Design

%% Communication Settings
KcList = 20 : 10 : 40;
M = 16;
L = 20;
Pt = 1;
N0dB = -10;
N0 = 10^(N0dB / 10);

%% Radar Settings
d = 0.5;
lambda = 2 * d;
pos = (0:M - 1) * d;
normalizedPos = pos / lambda;

theta_est = [-60, 0, 60];
Delta = 5;
theta_grid = -90:1:90;
P_des = zeros(size(theta_grid));

idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - Delta & theta_grid <= theta_est(i) + Delta;
end
P_des(idx) = 1;

%% Directional Beampattern
DirectRd_raw = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd = DirectRd_raw / M;
DirectRd = (DirectRd + DirectRd') / 2;

%% Tradeoff Settings
rhodB = [-30, -25, -20, -15, -10, -8, -6, -4, -2, -1];
rhoList = 10.^(rhodB ./ 10);

%% Simulation Settings
Iters = 1000;

DirectRateArray = zeros(Iters, length(rhoList), length(KcList));
DirectTradeoffBPArray = zeros(Iters, length(rhoList), length(KcList));

%% Monte Carlo Simulation
for iter = 1:Iters
    for idxRho = 1:length(rhoList)
        rho = rhoList(idxRho);
        for idxKc = 1:length(KcList)
            Kc = KcList(idxKc);
            H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);
            data = randi([0, 3], Kc, L);
            S = pskmod(data, 4, pi / 4, 'gray');
            
            DirectStrictX = strict_waveform(H, S, DirectRd, L);
            DirectTradeoffX0 = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);
            
            DirectTradeoffX = sqrt(M) * DirectTradeoffX0;
            
            DirectRateArray(iter, idxRho, idxKc) = average_user_rate(H, DirectTradeoffX / sqrt(M), S, N0);
            DirectTradeoffBPArray(iter, idxRho, idxKc) = radar_beampattern_mse_fig5(DirectRd, DirectTradeoffX, theta_grid, M);
        end
    end
    clc;
    disp(['Progress - ', num2str(iter), '/', num2str(Iters)]);
end

%% Average Results
DirectRate = squeeze(mean(real(DirectRateArray), 1));

DirectRate1 = DirectRate(:, 1);
DirectRate2 = DirectRate(:, 2);
DirectRate3 = DirectRate(:, 3);

DirectTradeoffBP = real(DirectTradeoffBPArray(1, :, :));
DirectTradeoffBP = 10 * log10(DirectTradeoffBP);

DirectTradeoffBP1 = squeeze(DirectTradeoffBP(:, :, 1)).';
DirectTradeoffBP2 = squeeze(DirectTradeoffBP(:, :, 2)).';
DirectTradeoffBP3 = squeeze(DirectTradeoffBP(:, :, 3)).';

%% Figure 5
figure;
plot(DirectRate1, DirectTradeoffBP1, 'b', 'LineWidth', 1.5);
hold on;
plot(DirectRate2, DirectTradeoffBP2, 'k', 'LineWidth', 1.5);
plot(DirectRate3, DirectTradeoffBP3, 'r', 'LineWidth', 1.5);
grid on;
xlabel('Average Achievable Rate (bps/Hz/user)');
ylabel('Average MSE (dB)');
legend('K=20', 'K=30', 'K=40', 'Location', 'southeast');

%% Local Function
function mseValue = radar_beampattern_mse_fig5(DirectRd, DirectXTradeoff, theta_grid, M)
DBP = zeros(length(theta_grid), 1);
DTBP = zeros(length(theta_grid), 1);

for idxTheta = 1:length(theta_grid)
    theta = theta_grid(idxTheta);
    a = exp(1j * pi * (0:M - 1)' * sind(theta));
    
    DBP(idxTheta) = real(a' * (DirectRd * DirectRd') * a) / real(trace(DirectRd * DirectRd'));
    DTBP(idxTheta) = real(a' * (DirectXTradeoff * DirectXTradeoff') * a) / real(trace(DirectXTradeoff * DirectXTradeoff'));
end

mseValue = norm(DTBP - DBP, 2)^2;
end