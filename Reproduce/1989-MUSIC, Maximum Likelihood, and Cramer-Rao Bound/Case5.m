%  第五种：原文中的复正弦频率闭式 CRB：估计 $\omega$

clear;
clc;
close all;
rng(42);
M = 10;
T = 200;
P = 1;
MC = 1000;
omega0 = 0.4 * pi;
omega_grid = linspace(0.01 * pi, 0.99 * pi, 4000);
SNRdB = -10:2:30;
MSE_music = zeros(length(SNRdB), 1);
CRB_theory = zeros(length(SNRdB), 1);
a0 = steering_vec_freq(M, omega0);
for isnr = 1:length(SNRdB)
    snr = 10^(SNRdB(isnr) / 10);
    sigma2 = P / snr;
    CRB_theory(isnr) = 6 * sigma2 / (T * P * M * (M^2 - 1));
    err = zeros(MC, 1);
    for imc = 1:MC
        x = sqrt(P / 2) * (randn(1, T) + 1j * randn(1, T));
        Y0 = a0 * x;
        noise = sqrt(sigma2 / 2) * (randn(M, T) + 1j * randn(M, T));
        Y = Y0 + noise;
        omega_hat = music_freq_1source(Y, omega_grid);
        err(imc) = omega_hat - omega0;
    end
    MSE_music(isnr) = mean(abs(err).^2);
end
figure;
semilogy(SNRdB, CRB_theory, 'k-o', 'LineWidth', 1.5);
hold on;
grid on;
semilogy(SNRdB, MSE_music, 'r-s', 'LineWidth', 1.5);
xlabel('SNR / dB');
ylabel('MSE / (rad/sample)^2');
legend('Closed-form CRB for \omega', 'MUSIC MSE', 'Location', 'southwest');
title('Single complex sinusoid frequency estimation');
function a = steering_vec_freq(M, omega)
    k = (0:M-1).';
    a = exp(1j * k * omega);
end
function omega_hat = music_freq_1source(Y, omega_grid)
    M = size(Y, 1);
    T = size(Y, 2);
    Rhat = Y * Y' / T;
    [V, Lambda] = eig(Rhat);
    lambda = real(diag(Lambda));
    [~, idx] = sort(lambda, 'descend');
    V = V(:, idx);
    En = V(:, 2:M);
    Pmu = zeros(length(omega_grid), 1);
    for ig = 1:length(omega_grid)
        a = exp(1j * (0:M-1).' * omega_grid(ig));
        Pmu(ig) = 1 / real(a' * En * En' * a);
    end
    [~, idmax] = max(Pmu);
    omega_hat = omega_grid(idmax);
end