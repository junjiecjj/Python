
%  Case3 第三种：单目标、有限快拍 CRB
 

clear;
clc;
close all;
rng(42);
N = 64;
fc = 100e9;
lambda_c = 3e8 / fc;
d = lambda_c / 2;
T = 100;
Nit = 1000;
SNRdBs = -10:2:20;
theta = deg2rad(30);
H = genSteerVector(theta, N, d, lambda_c);
a1 = genPartialSteerVector(theta, N, d, lambda_c, 1);
MseMUSIC = zeros(size(SNRdBs));
MseESPRIT = zeros(size(SNRdBs));
MseESPRIT_tls = zeros(size(SNRdBs));
MseESPRIT_tls1 = zeros(size(SNRdBs));
MseESPRIT_ml = zeros(size(SNRdBs));
CRLB = zeros(size(SNRdBs));
D = a1;
Cst = D' * (eye(N) - H / (H' * H) * H') * D;
for i = 1:length(SNRdBs)
    snr_dB = SNRdBs(i);
    fprintf('==============================================\n');
    fprintf('SNR = %g dB\n', snr_dB);
    for it = 1:Nit
        if mod(it - 1, 10) == 0
            fprintf('%d/%d\n', it, Nit);
        end
        X = sqrt(1 / 2) * (randn(1, T) + 1j * randn(1, T));
        y = zeros(N, T);
        for t = 1:T
            tmp = H * X(1, t);
            sig_power = mean(abs(tmp).^2);
            noise_var = sig_power * 10^(-snr_dB / 10);
            noise = sqrt(noise_var / 2) * (randn(N, 1) + 1j * randn(N, 1));
            y(:, t) = tmp + noise;
        end
        theta_MUSIC = MUSIC_rad(y, 1);
        MseMUSIC(i) = MseMUSIC(i) + abs(theta_MUSIC - theta)^2;
        Rxx = y * y' / T;
        theta_ESPRIT = ESPRIT_rad(Rxx, 1, N);
        MseESPRIT(i) = MseESPRIT(i) + abs(theta_ESPRIT(1) - theta)^2;
        psi = TLS_ESPRIT_rad(y, 1);
        theta_ESPRIT_tls = log(psi) / (1j * pi);
        MseESPRIT_tls(i) = MseESPRIT_tls(i) + abs(asin(theta_ESPRIT_tls) - theta)^2;
        [DOA_esp_ml, DOA_esp_tls] = DOA_ESPRIT_rad(y, 1, 2, 1);
        MseESPRIT_tls1(i) = MseESPRIT_tls1(i) + abs(DOA_esp_tls(1) - theta)^2;
        MseESPRIT_ml(i) = MseESPRIT_ml(i) + abs(DOA_esp_ml(1) - theta)^2;
        sigma2 = 10^(-snr_dB / 10);
        CRLB(i) = CRLB(i) + sigma2 / 2 / real(Cst * (X * X'));
    end
end
MseMUSIC = MseMUSIC / Nit;
MseESPRIT = MseESPRIT / Nit;
MseESPRIT_tls = MseESPRIT_tls / Nit;
MseESPRIT_tls1 = MseESPRIT_tls1 / Nit;
MseESPRIT_ml = MseESPRIT_ml / Nit;
CRLB = CRLB / Nit;
figure;
semilogy(SNRdBs, MseMUSIC, '-s', 'LineWidth', 1.5);
hold on;
grid on;
semilogy(SNRdBs, MseESPRIT, '-*', 'LineWidth', 1.5);
semilogy(SNRdBs, MseESPRIT_tls, '-d', 'LineWidth', 1.5);
semilogy(SNRdBs, MseESPRIT_ml, '-^', 'LineWidth', 1.5);
semilogy(SNRdBs, MseESPRIT_tls1, '-v', 'LineWidth', 1.5);
semilogy(SNRdBs, CRLB, '--o', 'LineWidth', 1.5);
xlabel('SNR / dB');
ylabel('MSE / rad^2');
legend('MUSIC', 'ESPRIT', 'ESPRIT TLS', 'ESPRIT ML', 'ESPRIT TLS1', 'CRLB', 'Location', 'southwest');
title('MUSIC / ESPRIT MSE and CRLB versus SNR');
function a = genSteerVector(theta, N, d, lambda)
    n = (0:N-1).';
    a = exp(1j * 2 * pi * d * sin(theta) * n / lambda);
end
function a = genPartialSteerVector(theta, N, d, lambda, flag)
    n = (0:N-1).';
    base = exp(1j * 2 * pi * d * sin(theta) * n / lambda);
    if flag == 1
        a = 1j * 2 * pi * d * n * cos(theta) / lambda .* base;
    else
        a = -(2 * pi * d * n * cos(theta) / lambda).^2 .* base - 1j * 2 * pi * d * n * sin(theta) / lambda .* base;
    end
end
function theta_hat = MUSIC_rad(y, K)
    [N, ~] = size(y);
    Range = pi;
    thre = 1e-12;
    center = 0;
    nit = 20;
    Rxx = y * y';
    [V, D] = eig(Rxx);
    lambda = real(diag(D));
    [~, idx] = sort(lambda, 'descend');
    V = V(:, idx);
    Un = V(:, K+1:N);
    UnUnH = Un * Un';
    while Range > thre
        theta_grid = linspace(center - Range, center + Range, nit);
        Pmu = zeros(nit, 1);
        for i = 1:nit
            ang = theta_grid(i);
            a = exp(1j * pi * (0:N-1).' * sin(ang));
            Pmu(i) = 1 / abs(a' * UnUnH * a);
        end
        [~, idmax] = max(abs(Pmu));
        center = center - Range + 2 * Range * (idmax - 1) / (nit - 1);
        Range = Range / 10;
    end
    theta_hat = center;
end
function value = TLS_ESPRIT_rad(y, L)
    K_sub = size(y, 1);
    Y_sub1 = y(1:end-1, :);
    Y_sub2 = y(2:end, :);
    Z_mtx = [Y_sub1; Y_sub2];
    R_ZZ = Z_mtx * Z_mtx';
    [U, ~, ~] = svd(R_ZZ);
    Es = U(:, 1);
    Es = Es(:);
    Exy = [Es(1:K_sub-1), Es(K_sub:end)];
    EE = Exy' * Exy;
    [~, ~, V] = svd(EE);
    EE = V;
    E12 = EE(1:L, L+1:2*L);
    E22 = EE(L+1:2*L, L+1:2*L);
    Psi = -E12 / E22;
    value = eig(Psi);
    value = value(1);
end
function Theta = ESPRIT_rad(Rxx, K, N)
    [U, D] = eig(Rxx);
    lambda = real(diag(D));
    [~, idx] = sort(lambda, 'descend');
    U = U(:, idx);
    Us = U(:, 1:K);
    Ux = Us(1:K, :);
    Uy = Us(2:K+1, :);
    Uxy = [Ux, Uy];
    Uxy = Uxy' * Uxy;
    [V, D] = eig(Uxy);
    lambda = real(diag(D));
    [~, idx] = sort(lambda, 'descend');
    V = V(:, idx);
    F0 = V(1:K, K+1:2*K);
    F1 = V(K+1:2*K, K+1:2*K);
    Psi = -F0 / F1;
    ev = eig(Psi);
    Theta = asin(angle(ev) / pi);
    Theta = sort(real(Theta));
end
function [DOA_esp_ml, DOA_esp_tls] = DOA_ESPRIT_rad(X, K, lambda, d)
    N = size(X, 1);
    x_esp = [X(1:N-1, :); X(2:N, :)];
    R_esp = cov(x_esp');
    [W, D] = eig(R_esp);
    lambda_eig = real(diag(D));
    [~, idx] = sort(lambda_eig, 'descend');
    W = W(:, idx);
    U_s = W(:, 1:K);
    U_s1 = U_s(1:N-1, :);
    U_s2 = U_s(N:end, :);
    mat_esp_ml = pinv(U_s1) * U_s2;
    ev_ml = eig(mat_esp_ml);
    DOA_esp_ml = asin(angle(ev_ml) * lambda / (2 * pi * d));
    Us12 = [U_s1, U_s2];
    [~, ~, V] = svd(Us12);
    E12 = V(1:K, K+1:end);
    E22 = V(K+1:end, K+1:end);
    mat_esp_tls = -E12 / E22;
    ev_tls = eig(mat_esp_tls);
    DOA_esp_tls = asin(angle(ev_tls) * lambda / (2 * pi * d));
end

