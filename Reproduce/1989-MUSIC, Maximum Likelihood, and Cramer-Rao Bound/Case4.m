%  第四种：单目标、大快拍平均功率 CRB：估计物理角度 $\theta$

clear;
clc;
close all;
rng(42);
M = 10;
T = 200;
P = 1;
MC = 10;
theta0 = deg2rad(30);
theta_grid = deg2rad(-60:0.02:60);
SNRdB = -10:2:30;
MSE_music = zeros(length(SNRdB), 1);
CRB_theory = zeros(length(SNRdB), 1);
a0 = steering_vec_angle(M, theta0);
d0 = steering_derivative_angle_single(M, theta0);
Pi_perp = eye(M) - a0 / (a0' * a0) * a0';
h0 = real(d0' * Pi_perp * d0);
for isnr = 1:length(SNRdB)
    snr = 10^(SNRdB(isnr) / 10);
    sigma2 = P / snr;
    CRB_theory(isnr) = sigma2 / (2 * T * P * h0);
    err = zeros(MC, 1);
    for imc = 1:MC
        x = sqrt(P / 2) * (randn(1, T) + 1j * randn(1, T));
        Y0 = a0 * x;
        noise = sqrt(sigma2 / 2) * (randn(M, T) + 1j * randn(M, T));
        Y = Y0 + noise;
        theta_hat = music_doa_1source(Y, theta_grid);
        err(imc) = theta_hat - theta0;
    end
    MSE_music(isnr) = mean(abs(err).^2);
end
figure;
semilogy(SNRdB, CRB_theory, 'k-o', 'LineWidth', 1.5);
hold on;
grid on;
semilogy(SNRdB, MSE_music, 'r-s', 'LineWidth', 1.5);
xlabel('SNR / dB');
ylabel('MSE / rad^2');
legend('Single-source average-power CRB', 'MUSIC MSE', 'Location', 'southwest');
title('Single-source DOA CRB and MUSIC MSE');
function a = steering_vec_angle(M, theta)
    k = (0:M-1).';
    a = exp(1j * pi * k * sin(theta));
end
function d = steering_derivative_angle_single(M, theta)
    k = (0:M-1).';
    a = exp(1j * pi * k * sin(theta));
    d = 1j * pi * k * cos(theta) .* a;
end
function theta_hat = music_doa_1source(Y, theta_grid)
    M = size(Y, 1);
    T = size(Y, 2);
    Rhat = Y * Y' / T;
    [V, Lambda] = eig(Rhat);
    lambda = real(diag(Lambda));
    [~, idx] = sort(lambda, 'descend');
    V = V(:, idx);
    En = V(:, 2:M);
    Pmu = zeros(length(theta_grid), 1);
    for ig = 1:length(theta_grid)
        a = exp(1j * pi * (0:M-1).' * sin(theta_grid(ig)));
        Pmu(ig) = 1 / real(a' * En * En' * a);
    end
    [~, idmax] = max(Pmu);
    theta_hat = theta_grid(idmax);
end