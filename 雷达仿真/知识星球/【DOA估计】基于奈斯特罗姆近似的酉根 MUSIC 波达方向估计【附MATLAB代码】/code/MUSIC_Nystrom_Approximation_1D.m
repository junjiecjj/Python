clc;
clear;
close all;
tic;
samples=1;%no of samples

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
%R_xx = (X * X') / K; % Sample covariance matrix

%% Nyström Approximation: Random Sensor Selection
Na = 10; % Number of randomly selected sensors (Na < L)
indices = randperm(L, Na); % Randomly choose Na sensors
Y = X(indices, :); % Selected sensor outputs

%% Covariance Matrix R_yy for Nyström
A_y = A(indices, :); % Subset of the direction matrix
R_yy = (Y * Y') / K; % Sample covariance from selected sensors

%% Cross-Covariance Matrix R_xy
R_xy = (X * Y') / K;

%% EVD of R_yy
[U_y, D_y] = eig(R_yy);
[~, idx] = sort(diag(D_y), 'descend');
U_y = U_y(:, idx);
D_y = D_y(idx, idx);

%% Compute Nyström Eigenvectors
lambda_y = D_y(1:M, 1:M); % Largest eigenvalues
U_ns=zeros(L,M);
for i=1:M
U_ns(:,i) = (1 / (lambda_y(i,i))) * R_xy * U_y(:, i);
end
%% Noise Subspace Estimation using Nyström
U_ns_orth = U_ns; % Noise subspace

%% MUSIC Spectrum Calculation
theta_scan = -90:0.1:90;
p_theta = zeros(1, length(theta_scan));
for i = 1:length(theta_scan)
    gamma_scan = -2*pi*d*sin(deg2rad(theta_scan(i)));
    a_theta = exp(1j * (0:L-1)' * gamma_scan);
    p_theta(i) = 1 / (a_theta' * (eye(L) - U_ns_orth * U_ns_orth') * a_theta);
end

%% Plot the MUSIC Spectrum
 p_theta_dB = 10*log10(abs(p_theta) / max(abs(p_theta)));

%% Find DOA Peaks
[peak_values, peak_indices] = findpeaks(p_theta_dB, 'SortStr', 'descend');
estimated_DOAs = theta_scan(peak_indices(1:2*M));
estimated_DOAs=sort(estimated_DOAs,'ascend');
estimated_DOAs_final=zeros(M,1);
for i=1:M
    estimated_DOAs_final(i)=0.5*(estimated_DOAs(2*i-1)+estimated_DOAs(2*i));
end
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
