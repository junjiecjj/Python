clear;
clc;
close all;
M = 10;
N = 200;
P = 1;
theta0 = 10;
SNRdB = -30:2:10;
MC = 100;
theta_grid = -60:0.01:60;
num_snr = length(SNRdB);
MSE_music = zeros(num_snr, 1);
CRB_theta = zeros(num_snr, 1);
a0 = steering_vec(M, theta0);
d0 = steering_derivative_deg(M, theta0);
Pi_perp = eye(M) - a0 / (a0' * a0) * a0';
h0 = real(d0' * Pi_perp * d0);
for isnr = 1:num_snr
    snr = 10^(SNRdB(isnr) / 10);
    sigma = P / snr;
    CRB_theta(isnr) = sigma / (2 * N * P * h0);
    err = zeros(MC, 1);
    for imc = 1:MC
        s = sqrt(P) * exp(1j * 2 * pi * rand(1, N));
        noise = sqrt(sigma / 2) * (randn(M, N) + 1j * randn(M, N));
        Y = a0 * s + noise;
        theta_hat = music_doa_1source(Y, theta_grid);
        err(imc) = theta_hat - theta0;
    end
    MSE_music(isnr) = mean(err.^2);
end
figure;
semilogy(SNRdB, CRB_theta, 'k-o', 'LineWidth', 1.5);
hold on;
grid on;
semilogy(SNRdB, MSE_music, 'r-s', 'LineWidth', 1.5);
xlabel('SNR / dB');
ylabel('MSE / deg^2');
legend('Theoretical CRB', 'MUSIC Monte Carlo MSE', 'Location', 'southwest');
title(['Single-source DOA estimation, M = ', num2str(M), ', N = ', num2str(N)]);
function theta_hat = music_doa_1source(Y, theta_grid)
    M = size(Y, 1);
    N = size(Y, 2);
    Rhat = Y * Y' / N;
    [V, Lambda] = eig(Rhat);
    lambda = real(diag(Lambda));
    [~, idx] = sort(lambda, 'ascend');
    En = V(:, idx(1:M-1));
    Pmu = zeros(length(theta_grid), 1);
    for ig = 1:length(theta_grid)
        a = steering_vec(M, theta_grid(ig));
        denom = real(a' * En * En' * a);
        Pmu(ig) = 1 / denom;
    end
    [~, idmax] = max(Pmu);
    theta_hat = theta_grid(idmax);
end
function a = steering_vec(M, theta_deg)
    k = (0:M-1).';
    theta_rad = theta_deg * pi / 180;
    a = exp(1j * pi * k * sin(theta_rad));
end
function d = steering_derivative_deg(M, theta_deg)
    k = (0:M-1).';
    theta_rad = theta_deg * pi / 180;
    a = exp(1j * pi * k * sin(theta_rad));
    d = 1j * pi * k * cos(theta_rad) * (pi / 180) .* a;
end