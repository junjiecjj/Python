clc;
clear;
close all;
tic;
samples=1; % no of samples

%% Parameters
M = 2;                          % Number of sources
N = 10;                          % Number of sensors on each side, so total L = 2N + 1
L = 2*N + 1;                    % Total number of sensors
d = 0.5;                        % Element spacing (in wavelength units)
SNR_dB = 10;                    % Signal-to-noise ratio in dB
SNR = 10^(SNR_dB/10);           % Linear SNR
theta_deg = [-45, 45];          % True DOAs in degrees
theta_rad = deg2rad(theta_deg); % Convert angles to radians
lambda = 1;                     % Wavelength
sigma2 = 1/SNR;                 % Noise variance
K = 200;                        % Number of snapshots

%% Steering Matrix Generation
A = zeros(L, M);
for m = 1:M
    gamma_m = -2*pi*d*sin(theta_rad(m));
    A(:, m) = exp(1j * (0:L-1)' * gamma_m);
end

%% Generate Signals and Noise
S = (randn(M, K) + 1j * randn(M, K)) / sqrt(2); % Complex Gaussian source signals
w = sqrt(sigma2) * (randn(L, K) + 1j * randn(L, K)) / sqrt(2); % Noise matrix
X = A * S + w; % Received signal matrix

%% Covariance Matrix Estimation
R_xx = (X * X') / K; % Sample covariance matrix
% Eigen-decomposition
[E, D] = eig(R_xx);
[~, idx] = sort(diag(D), 'descend');  
En = E(:, idx(M+1:end)); % Noise subspace

%% MUSIC Spectrum Calculation
theta_scan = -90:0.1:90;
p_theta = zeros(1, length(theta_scan));
for i = 1:length(theta_scan)
    gamma_scan = -2*pi*d*sin(deg2rad(theta_scan(i)));
    a_theta = exp(1j * (0:L-1)' * gamma_scan);
    p_theta(i) = 1 / (a_theta' * (En*En') * a_theta);
end

%% Plot the MUSIC Spectrum
 p_theta_dB = 10*log10(abs(p_theta) / max(abs(p_theta)));

%% Find DOA Peaks
[peak_values, peak_indices] = findpeaks(p_theta_dB, 'SortStr', 'descend');
estimated_DOAs = theta_scan(peak_indices(1:M));
estimated_DOAs_final=sort(estimated_DOAs,'ascend');
disp("Estimated DoAs are:");
disp(estimated_DOAs_final);
toc;
%% Highlight Detected DOAs on the Plot
figure;
plot(theta_scan, p_theta_dB, 'b', 'LineWidth', 1.5); % Original spectrum
hold on;
plot(estimated_DOAs_final, interp1(theta_scan, p_theta_dB, estimated_DOAs_final), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
xlabel('Angle (degrees)');
ylabel('Spatial Spectrum (dB)');
title('Modified MUSIC Spectrum with Nyström Approximation');
legend('MUSIC Spectrum', 'Refined DOA Estimates');
grid on;
