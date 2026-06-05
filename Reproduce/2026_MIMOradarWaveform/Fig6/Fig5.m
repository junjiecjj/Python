

clc;
clear all;
close all;
addpath('./functions_2018ICC');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');
rng(42);

%% Figure 5: Trade-off of Directional Beampattern Design

%% Communication Settings
KcList = 6 : 2 : 10;
M = 16;
L = 100;
Pt = 1;
c = ones(M, 1) * Pt/M;       % 对角元固定值

% comm snr
SNRdB = 10;
N0 = Pt / 10^(SNRdB/10);

%% Radar Settings
d = 0.5;
lambda = 2 * d;
pos = (0:M - 1) * d;
normalizedPos = pos / lambda;
afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1
%% Desired Beampattern
theta_est = [0];             % 目标角度估计（度）
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

A = steeringMatrixULA1D(normalizedPos, theta_grid);

%% Directional Beampattern
%  文献1：On Probing Signal Design For MIMO Radar, C. Beampattern Matching Design
%  diag(R)=1/M, trace(R)=1, wc=0
w_l = ones(length(theta_grid), 1);
w_c = 0;
[DirectRd1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);
fprintf('trace(DirectRd1) = %.6f\n',  trace(DirectRd1));
P_des1 = P_des * alpha1;

%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  helperMMSECovariance 默认 diag(R)=1, trace(R)=M, 为了和文献1对齐，将 R 除以 M，使 trace(R)=1
DirectRd2_raw = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2_raw / M;
DirectRd2 = projectToPSD(DirectRd2);
DirectRd2 = DirectRd2 + 1e-10 * eye(size(M));
fprintf('trace(DirectRd2) = %.6f\n',  trace(DirectRd2));

[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt); 
fprintf('trace(DirectRd3) = %.6f\n',  trace(DirectRd3));

DirectRd = DirectRd2;
%% Tradeoff Settings
% rhoList = 0.1:0.1:0.9;
rhodB = [-30 -25 -20 -15 -10 -8 -6 -4 -2 -1, -0.06];
rhoList = 10.^(rhodB ./ 10);


%% Simulation Settings
Iters = 1000;

DirectRateArray = zeros(Iters, length(rhoList), length(KcList));
DirectTradeoffBPArray = zeros(Iters, length(rhoList), length(KcList));

%% Monte Carlo Simulation
for iter = 1:Iters
    clc; disp(['Progress - ', num2str(iter), '/', num2str(Iters)]);
    for idxRho = 1:length(rhoList)
        rho = rhoList(idxRho);
        for idxKc = 1:length(KcList)
            Kc = KcList(idxKc);
            H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);
            data = randi([0, 3], Kc, L);
            S = pskmod(data, 4, pi / 4, 'gray');
            
            DirectStrictX = strict_waveform(H, S, DirectRd, L);
            DirectTradeoffX = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);
            
            % DirectTradeoffX = sqrt(M) * DirectTradeoffX0;
            Rtmp = DirectTradeoffX * DirectTradeoffX' / L;
            
            DirectRateArray(iter, idxRho, idxKc) = average_user_rate(H, DirectTradeoffX, S, N0);
            DirectTradeoffBPArray(iter, idxRho, idxKc) = sqrt(norm(diag(A'*DirectRd*A) - diag(A'*Rtmp*A), 2));
            % radar_beampattern_mse_fig5(DirectRd, DirectTradeoffX, theta_grid, M);
        end
    end
end

%% Average Results
DirectRate = squeeze(mean(real(DirectRateArray), 1));

DirectRate1 = DirectRate(:, 1);
DirectRate2 = DirectRate(:, 2);
DirectRate3 = DirectRate(:, 3);

DirectTradeoffBP = squeeze(mean(real(DirectTradeoffBPArray), 1));
DirectTradeoffBP = 10 * log10(DirectTradeoffBP);

DirectTradeoffBP1 = squeeze(DirectTradeoffBP(:, 1)).';
DirectTradeoffBP2 = squeeze(DirectTradeoffBP(:, 2)).';
DirectTradeoffBP3 = squeeze(DirectTradeoffBP(:, 3)).';

%% Figure 5
figure(2);
plot(DirectRate1, DirectTradeoffBP1, 'b', 'LineWidth', 1.5); hold on;
plot(DirectRate2, DirectTradeoffBP2, 'k', 'LineWidth', 1.5); hold on;
plot(DirectRate3, DirectTradeoffBP3, 'r', 'LineWidth', 1.5);
grid on;
xlabel('Average Achievable Rate (bps/Hz/user)');
ylabel('Average MSE (dB)');
legend('K=6', 'K=8', 'K=10', 'Location', 'southeast');


if 1
    %% 计算 beampattern
    P_lit1 = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        ai = afun(theta_grid(i));
        P_lit1(i) = real(ai' * DirectRd * ai);
    end

    figure(1);
    plot(theta_grid, 10 * log10(P_des1 + eps), 'k--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, 10 * log10(P_lit1 + eps), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('\theta (degrees)');
    ylabel('Beampattern');
    legend('Desired', 'Beampattern Matching', 'Squared Error', 'my Squared Error', 'Location', 'best');
    title('Comparison under trace(R)=1');
    xlim([-90, 90]);
    ylim([-40, 20]);
end


%% Local Function
function mseValue = radar_beampattern_mse_fig5(DirectRd, DirectXTradeoff, theta_grid, M)
    DBP = zeros(length(theta_grid), 1);
    DTBP = zeros(length(theta_grid), 1);
    L = size(DirectXTradeoff, 2);
    Rtmp = DirectXTradeoff * DirectXTradeoff'/L;
    for idxTheta = 1:length(theta_grid)
        theta = theta_grid(idxTheta);
        a = exp(1j * pi * (0:M - 1)' * sind(theta));
        
        DBP(idxTheta) = a' * DirectRd * a;
        DTBP(idxTheta) = a' * Rtmp * a;
    end
    
    mseValue = norm(DTBP - DBP, 2)^2;
end


function mseValue = radar_beampattern_mse_fig5x(DirectRd, DirectXTradeoff, theta_grid, M)
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