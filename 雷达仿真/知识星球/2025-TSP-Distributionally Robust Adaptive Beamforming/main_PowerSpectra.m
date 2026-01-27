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

% For reproduciability; year of 2025!
% rng(2025);    

% Signals, their DOAs, and their radiation powers
Theta = [-30 -22 30]*pi/180;    % Degrees
Ps    = [1 1 1];                % The first is the signal, the remainings are interference
K     = length(Theta);          % Actual number of targets     (ground truth)
KK    = 3;                      % Nominal number of targets    (user's belief, not necessarily equal to ground truth; K <= KK <= N - 1)

% Number of antennas
N = 10;

% Grid Search for DOA
ThetaSweep = -pi/2:0.001:pi/2;
ThetaLen   = length(ThetaSweep);

% SNR and noise power
SNR = 10^(10/10);               % 10dB
Pv = Ps(1)/SNR;

% Nuber of snapshots
L = max(30, N+1);       % For good results, use 25, 50

% True Statistics
Rx0 = a0(N, Theta)*diag(Ps)*a0(N, Theta)' + Pv*eye(N);
[U0, D0] = eig(Rx0);
[d0, index0] = sort(diag(D0), 'descend');
D0 = diag(d0);
U0 = U0(:, index0);

Us0 = U0(:, 1:KK);
Uv0 = U0(:, KK + 1: end);

%% Find Real Positions of Impules in Spectra
IdealSpectra = zeros(ThetaLen, 0);
for k = 1:K
    theta = Theta(k);
    tempArray = abs(ThetaSweep - theta);
    [~, ind] = min(tempArray);
    IdealSpectra(ind) = 1;
end

%% Output power spectra
% Under True Rx0
SpectraCapon0   = zeros(ThetaLen, 1);               % Capon with Rx0
SpectraMusic0   = zeros(ThetaLen, 1);               % Music with Rx0

% True Spectra
for i = 1:ThetaLen
    SpectraCapon0(i)  = 1 / abs(a0(N, ThetaSweep(i))'* (Rx0)^-1 *a0(N, ThetaSweep(i)));
    SpectraMusic0(i)  = 1 / (abs(a0(N, ThetaSweep(i))'* (Uv0*Uv0') *a0(N, ThetaSweep(i))) + 1e-5);
end

% Under Nominal Rx
MonteCarlo        = 1;                               % Monte-Carlo Trials -- Use 500

SpectraCapon      = zeros(ThetaLen, MonteCarlo);      % Capon with Rx
SpectraCaponDL    = zeros(ThetaLen, MonteCarlo);      % Capon with Rx and diagonal loading            (DL)
SpectraCaponUDL   = zeros(ThetaLen, MonteCarlo);      % Capon with Rx and unbalanced diagonal loading (UDL)
SpectraMaxEnt     = zeros(ThetaLen, MonteCarlo);      % MaxEnt spectra
SpectraMusic      = zeros(ThetaLen, MonteCarlo);      % MUSIC pseudo-spectra

% Nominal Spectra
for mc = 1:MonteCarlo
    % Radiation signals
    S = diag(sqrt(Ps)) * sqrt(0.5) * (randn(K, L) + 1j*randn(K, L));
    % Channel noise
    V = sqrt(Pv)  * sqrt(0.5) * (randn(N, L) + 1j*randn(N, L));
    % Signal transmission abd reception
    X = a0(N, Theta)*S + V;
    
    % Sample covariance and its eigendecomposition
    Rx = X*X'/L;
    [U, D] = eig(Rx);
    [d, index] = sort(diag(D), 'descend');
    D = diag(d);
    U = U(:, index);

    % Noise subspace bases
    Us = U(:, 1:KK);
    Ds = D(1:KK, 1:KK);
    Uv = U(:, KK + 1: end);
    Dv = D(KK+1:end, KK+1:end);

    error = 0.01*eye(N)*randn(N, 1);
    stVec = @(N, theta) a0(N, theta) + error;

    for i = 1:ThetaLen
        % Nominal Capon
        SpectraCapon(i, mc)    = 1 / abs(stVec(N, ThetaSweep(i))'* (Rx)^-1 *stVec(N, ThetaSweep(i)));

        % Diagonal-Loading Capon
        epsilon_1 = 0.01;          %
        SpectraCaponDL(i, mc)  = 1 / abs(stVec(N, ThetaSweep(i))'* (Rx + epsilon_1*eye(N))^-1 *stVec(N, ThetaSweep(i)));

        % Unbalanced Diagonal-Loading (UDL) Capon
        delta_1 = 10;
        delta_2 = 0;
        SpectraCaponUDL(i, mc) = 1 / abs(stVec(N, ThetaSweep(i))'* (Us * (Ds + delta_1*eye(KK))^-1 * Us' + Uv * (Dv + delta_2*eye(N-KK))^-1 * Uv') *stVec(N, ThetaSweep(i)));

        % Maximum-Entropy
        % Mohammadzadeh (2020). Maximum entropy-based interference-plus-noise covariance matrix reconstruction for robust adaptive beamforming. IEEE Signal Processing Letters, 27, 845-849
        %
        % Performance is NOT stable, Why? Because it is a baised estimator!
        % It may 1) generate wrong locations of interferers, or 2) estimate wrong maglitudes of interferers!
        % Be cautious using this method!
        %
        u = [1; zeros(N-1, 1)];
        SpectraMaxEnt(i, mc)   =  1 / (abs(stVec(N, ThetaSweep(i))'* (Rx)^-1 * (u*u') * ((Rx)^-1)' *stVec(N, ThetaSweep(i))));

        % MUSIC --- Not really power spectra
        SpectraMusic(i, mc)    = 1 / abs(stVec(N, ThetaSweep(i))'* (Uv * Uv') *stVec(N, ThetaSweep(i)));
    end

    if mod(mc, 10) == 0
        clc;
        disp(['Monte-Carlo Trial: ' num2str(mc) '/' num2str(MonteCarlo)]);
    end
end

%%
PowerSpectra_Plot;






