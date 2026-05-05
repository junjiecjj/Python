% fig8.m: Fig. 8 in Xu, Stoica, Li 2006
% MSE vs subvector length M for beta3 and beta1, L0=32, sigma_e^2 fixed

clear; clc; close all;

L0 = 32;
f = [0.10, 0.11, 0.30];
beta_true = [exp(1j*pi/4); exp(1j*pi/3); exp(1j*pi/4)];
N = length(f);
ar_coef = 0.09;
sigma2_e = 0.01;   % innovation variance (fixed, as in paper Fig.8)

% M values to test (from 3 to 16 for MAFI, but AML can start from 1)
M_vals = 3:16;
nMC = 500;

MSE_AML_beta3 = zeros(size(M_vals));
MSE_AML_beta1 = zeros(size(M_vals));
MSE_MAFI_beta3 = zeros(size(M_vals));
MSE_MAFI_beta1 = zeros(size(M_vals));

for m_idx = 1:length(M_vals)
    M = M_vals(m_idx);
    L = L0 - M + 1;
    if L < N, continue; end   % need enough snapshots
    
    % Build steering matrix A (M x N)
    A = exp(1j * (0:M-1)' * (2*pi*f));
    % Build waveform matrix S (N x L)
    l_idx = 0:L-1;
    S = exp(1j * (2*pi*f') * l_idx);
    
    mse_aml3 = 0; mse_mafi3 = 0;
    mse_aml1 = 0; mse_mafi1 = 0;
    
    for mc = 1:nMC
        % Generate AR noise with fixed sigma2_e
        e = sqrt(sigma2_e/2) * (randn(1, L0) + 1j*randn(1, L0));
        z = filter(1, [1, -ar_coef], e);
        % Clean signal
        s_clean = zeros(1, L0);
        for n = 1:N
            s_clean = s_clean + beta_true(n) * exp(1j*2*pi*f(n)*(0:L0-1));
        end
        x = s_clean + z;
        % Build Hankel matrix
        X = hankel(x(1:M), x(M:end));
        
        % AML
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12, T = T + 1e-6*eye(M); end
        Tinv = inv(T);
        beta_AML = inv((A'*Tinv*A) .* (S*S').') * diag(A'*Tinv*X*S');
        mse_aml3 = mse_aml3 + norm(beta_AML(3) - beta_true(3))^2;
        mse_aml1 = mse_aml1 + norm(beta_AML(1) - beta_true(1))^2;
        
        % MAFI
        R_hat = T / (L - rank(S));
        if rcond(R_hat) < 1e-12, R_hat = R_hat + 1e-6*eye(M); end
        Rinv = inv(R_hat);
        beta_mafi = zeros(N,1);
        for n = 1:N
            a_n = exp(1j*2*pi*f(n)*(0:M-1)');
            temp = 0;
            for l = 1:L
                x_l = X(:,l);
                temp = temp + (a_n' * Rinv * x_l) / (a_n' * Rinv * a_n);
            end
            beta_mafi(n) = temp / L;
        end
        mse_mafi3 = mse_mafi3 + norm(beta_mafi(3) - beta_true(3))^2;
        mse_mafi1 = mse_mafi1 + norm(beta_mafi(1) - beta_true(1))^2;
    end
    MSE_AML_beta3(m_idx) = mse_aml3 / nMC;
    MSE_AML_beta1(m_idx) = mse_aml1 / nMC;
    MSE_MAFI_beta3(m_idx) = mse_mafi3 / nMC;
    MSE_MAFI_beta1(m_idx) = mse_mafi1 / nMC;
end

% Plot Fig.8
figure('Position', [100, 100, 1000, 400]);
subplot(1,2,1);
semilogy(M_vals, MSE_AML_beta3, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(M_vals, MSE_MAFI_beta3, 'bs-', 'LineWidth', 1.5);
xlabel('Subvector length M'); ylabel('MSE');
legend('AML', 'MAFI', 'Location', 'best');
title('(a) For β_3');
grid on;

subplot(1,2,2);
semilogy(M_vals, MSE_AML_beta1, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(M_vals, MSE_MAFI_beta1, 'bs-', 'LineWidth', 1.5);
xlabel('Subvector length M'); ylabel('MSE');
legend('AML', 'MAFI', 'Location', 'best');
title('(b) For β_1');
grid on;

sgtitle('Fig. 8: L_0 = 32, σ_e^2 = 0.01, colored AR(1) noise');