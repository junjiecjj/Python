%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}


clear;
clc;

addpath('./Utils/');

%% For reproduciability; year of 2025!
rng(2025);    

%% Consider error in steering vector?
isSteerVectorUncertain = 1;     % Do you think that the steer vector is uncertain? Yes (1) - It is Uncertatin; No (0) - It is Exact.
if isSteerVectorUncertain
    Delta = 0.01;
else
    Delta = 0;
end

%% Signals, their DOAs, and their radiation powers
Scenario = 0;
if Scenario == 0
    % close and strong interferers
    Theta = [-30 -22 30]*pi/180;    % DoA of signals and interferers (Degrees)
    Ps    = [1 10 10];              % The first is the signal, the remainings are interferences
elseif Scenario == 1
    % far and weak interferers
    Theta = [-30 0 30]*pi/180;      % DoA of signals and interferers (Degrees)
    Ps    = [1 1 1];                % The first is the signal, the remainings are interferences
else
    error('main_SINR :: Error in Scenario');
end

K     = length(Theta);          % Actual number of targets     (ground truth)
KK    = 3;                      % Nominal number of targets    (user's belief, not necessarily equal to ground truth; K <= KK <= N - 1)
                                % Try also KK = 5, KK = 7

%% Global Parameters
% Number of antennas
N = 10;                         % A typical value in literature
% Angles for estimating IPN matrix, leaving a [-G, +G] degree gap (centered at the signal's DoA) in the power spectra
G = 5;                          % This value was used in [Yujie Gu; TSP; 2012; "Robust Adaptive Beamforming Based on Interference Covariance Matrix Reconstruction and Steering Vector Estimation"]
ThetaIPN   = [-pi/2:0.01:(Theta(1) - G*pi/180), (Theta(1) + G*pi/180) :0.01:pi/2];  % The spacing is 0.57 degree (i.e., 0.01 radian)

%% Main Simulations
MonteCarlo = 500;               % Monte Carlo episode

isPlotAgainstSNR = 1;
if isPlotAgainstSNR
    % Nuber of snapshots
    Snapshot = max(30, N+1);    % 30， 80

    % SNR
    SNR = -20:30;

    % Processing
    SINR_vs_SNR;

else  % Against Snapshot
    % Nuber of snapshots
    Snapshot = N+1 : 100;

    % SNR
    SNR = 25;   % 10, 25 （dB）

    % Processing
    SINR_vs_Snapshot;
end

%% Plot Output SINR Results
SINR_Plot;






