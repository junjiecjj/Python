
% 2. 第二种：大快拍数近似 CRB：多目标、用协方差矩阵 $P$ 计算 CRB
clear;
clc;
close all;
rng(42);
M = 10;
K = 2;
T = 500;
MC = 100;
rho = 0.5;
Pmat = [1, rho; rho, 1];
theta_true = deg2rad([-10, 20]);
theta_grid = deg2rad(-60:0.05:60);
SNRdB = -10:2:30;
MSE_music = zeros(length(SNRdB), 1);
CRB_theory = zeros(length(SNRdB), 1);
A0 = steering_matrix_angle(M, theta_true);
D0 = steering_derivative_angle(M, theta_true);
Pi_perp = eye(M) - A0 / (A0' * A0) * A0';
B = D0' * Pi_perp * D0;
J0 = real(B .* transpose(Pmat));
L = chol(Pmat, 'lower');
for isnr = 1:length(SNRdB)
    snr = 10^(SNRdB(isnr) / 10);
    sigma2 = 1 / snr;
    CRB = sigma2 / (2 * T) * (J0 \ eye(K));
    CRB_theory(isnr) = mean(real(diag(CRB)));
    mse_tmp = zeros(MC, 1);
    for imc = 1:MC
        W = sqrt(1 / 2) * (randn(K, T) + 1j * randn(K, T));
        X = L * W;
        Y0 = A0 * X;
        noise = sqrt(sigma2 / 2) * (randn(M, T) + 1j * randn(M, T));
        Y = Y0 + noise;
        theta_hat = music_doa_k(Y, K, theta_grid);
        theta_hat = sort(theta_hat(:).');
        err = theta_hat - theta_true;
        mse_tmp(imc) = mean(abs(err).^2);
    end
    MSE_music(isnr) = mean(mse_tmp);
end
figure;
semilogy(SNRdB, CRB_theory, 'k-o', 'LineWidth', 1.5);
hold on;
grid on;
semilogy(SNRdB, MSE_music, 'r-s', 'LineWidth', 1.5);
xlabel('SNR / dB');
ylabel('MSE / rad^2');
legend('Large-snapshot CRB', 'MUSIC MSE', 'Location', 'southwest');
title(['Large-snapshot CRB, K = 2, \rho = ', num2str(rho)]);
function A = steering_matrix_angle(M, theta)
    k = (0:M-1).';
    K = length(theta);
    A = zeros(M, K);
    for i = 1:K
        A(:, i) = exp(1j * pi * k * sin(theta(i)));
    end
end
function D = steering_derivative_angle(M, theta)
    k = (0:M-1).';
    K = length(theta);
    D = zeros(M, K);
    for i = 1:K
        a = exp(1j * pi * k * sin(theta(i)));
        D(:, i) = 1j * pi * k * cos(theta(i)) .* a;
    end
end
function theta_hat = music_doa_k(Y, K, theta_grid)
    M = size(Y, 1);
    T = size(Y, 2);
    Rhat = Y * Y' / T;
    [V, Lambda] = eig(Rhat);
    lambda = real(diag(Lambda));
    [~, idx] = sort(lambda, 'descend');
    V = V(:, idx);
    En = V(:, K+1:M);
    Pmu = zeros(length(theta_grid), 1);
    for ig = 1:length(theta_grid)
        a = exp(1j * pi * (0:M-1).' * sin(theta_grid(ig)));
        Pmu(ig) = 1 / real(a' * En * En' * a);
    end
    theta_hat = pick_music_peaks(theta_grid, Pmu, K);
end
function theta_hat = pick_music_peaks(theta_grid, Pmu, K)
    step_deg = mean(diff(rad2deg(theta_grid)));
    min_sep = max(1, round(2 / step_deg));
    [~, order] = sort(Pmu, 'descend');
    chosen = [];
    for i = 1:length(order)
        idx = order(i);
        if isempty(chosen)
            chosen = idx;
        elseif all(abs(idx - chosen) >= min_sep)
            chosen = [chosen, idx];
        end
        if length(chosen) == K
            break;
        end
    end
    theta_hat = sort(theta_grid(chosen));
end