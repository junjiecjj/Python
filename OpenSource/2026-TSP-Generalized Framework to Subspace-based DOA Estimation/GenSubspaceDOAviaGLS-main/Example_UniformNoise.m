clear;
close all;
M=10;                    % Number of elements of the array
degsais = [34 36 19];    % True DOAs in Degree
psais = degsais*pi/180;  % True DOAs in Radian
K = length(psais);       % Number of sources
SNR_db = 20;             % SNR in dB
SNR = 10.^(SNR_db/10);
N = 10;                  % Number of snapshots

Qv = diag(ones(M,1));    % Actual Noise Covariance Matrix
eps = 1e-4;
%% Generating Received Signal
suminvQv = sum(1./diag(Qv));
sigmasq = (M/suminvQv)*SNR;    % Power of each source          

A = exp(-1j*pi*(0:M-1)'*sin(psais));     % Steering Matrix
S = sqrt(sigmasq/2)*(randn(K,N)+1j*randn(K,N));     % Source Signal
Noise = sqrt(1/2)*(Qv^(0.5))*(randn(M,N)+1j*randn(M,N));    % Noise Matrix
xmt = (A*S)+Noise;                  % Received Signal

R=(xmt*xmt')/N;
%% Estimating Noise Covariance Matrix
[Q] = ISB(R,K,M,eps);

%% Proposed Methods 
[PM_DOA_degree,PM_DOA_radian] = Proposed(R,K,M,5,xmt,[ M-1 :M ],Q);            % Proposed Method
[FB_PM_DOA_degree,FB_PM_DOA_radian] = FB_Proposed(R,K,M,5,xmt,[ M-1 :M ],Q,N);   % FB Proposed Method