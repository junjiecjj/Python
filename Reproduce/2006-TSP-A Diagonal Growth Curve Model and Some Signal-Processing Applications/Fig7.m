% fig7.m: Fig. 7 in Xu, Stoica, Li 2006 (Spectrum analysis)
% MSE vs local SNR for beta1 and beta3, L0=32, M=8

clear; clc; close all;

% Parameters
L0 = 32;                % total data length
M = 8;                  % subvector length (snapshot dimension)
L = L0 - M + 1;         % number of snapshots
f = [0.10, 0.11, 0.30]; % normalized frequencies
beta_true = [exp(1j*pi/4); exp(1j*pi/3); exp(1j*pi/4)];
N = length(f);

% Steering matrix A (M x N)
A = exp(1j * (0:M-1)' * (2*pi*f));

% Waveform matrix S (N x L) for each frequency
l_idx = 0:L-1;
S = exp(1j * (2*pi*f') * l_idx);

% AR coefficient for colored noise
ar_coef = 0.09;

% Local SNR definition from [28, Eq.(62)]
% It depends on the noise PSD at the frequency of interest.
% We will generate noise with different sigma_e^2 (innovation variance)
% and compute local SNR for each sinusoid.
% Local SNR for nth sinusoid: SNR_local_n = |beta_n|^2 / (2 * pi * S_z(omega_n))
% where S_z(omega) = sigma_e^2 / |1 - 0.09 e^{-j omega}|^2.

% Vector of local SNR values to simulate (dB)
local_SNR_dB_vals = -10:5:30;
nMC = 500;   % Monte Carlo runs

% Preallocate MSE matrices
MSE_AML_beta3 = zeros(size(local_SNR_dB_vals));
MSE_AML_beta1 = zeros(size(local_SNR_dB_vals));
MSE_MAFI_beta3 = zeros(size(local_SNR_dB_vals));
MSE_MAFI_beta1 = zeros(size(local_SNR_dB_vals));
CRB_beta3 = zeros(size(local_SNR_dB_vals));
CRB_beta1 = zeros(size(local_SNR_dB_vals));

for idx = 1:length(local_SNR_dB_vals)
    local_SNR_dB = local_SNR_dB_vals(idx);
    % Find sigma_e^2 such that the local SNR for beta3 (or beta1) equals local_SNR_dB
    % We will adjust sigma_e^2 for each sinusoid separately in the CRB calculation,
    % but in simulation the same noise realization must be used.
    % For fair comparison, we calibrate sigma_e^2 to achieve a given local SNR for the target sinusoid.
    % In Fig.7(a) it's for beta3, Fig.7(b) for beta1. We'll loop over two cases.
    
    % For simplicity, we run two separate simulations: one for beta3, one for beta1.
    % But to save time, we can compute required sigma_e^2 for each target.
    
    % Compute noise PSD at frequency f
    omega = 2*pi*f;
    PSD_factor = 1 ./ abs(1 - ar_coef*exp(-1j*omega)).^2;  % S_z(omega) / sigma_e^2
    
    % For beta3 (index 3)
    target_SNR_lin = 10^(local_SNR_dB/10);
    sigma2_e_beta3 = abs(beta_true(3))^2 / (2*pi * target_SNR_lin * PSD_factor(3));
    % For beta1 (index 1)
    sigma2_e_beta1 = abs(beta_true(1))^2 / (2*pi * target_SNR_lin * PSD_factor(1));
    
    % Use sigma2_e_beta3 for MC simulation when evaluating beta3 (subplot a)
    % and sigma2_e_beta1 for beta1 (subplot b). We'll run two separate MC loops.
    
    % === Subplot (a): beta3 ===
    sigma2_e = sigma2_e_beta3;
    mse_aml3 = 0; mse_mafi3 = 0;
    for mc = 1:nMC
        % Generate AR noise
        e = sqrt(sigma2_e/2) * (randn(1, L0) + 1j*randn(1, L0));
        z = filter(1, [1, -ar_coef], e);
        % Generate signal
        s_clean = zeros(1, L0);
        for n = 1:N
            s_clean = s_clean + beta_true(n) * exp(1j*2*pi*f(n)*(0:L0-1));
        end
        x = s_clean + z;
        % Build Hankel data matrix X (M x L)
        X = hankel(x(1:M), x(M:end));
        % AML estimation
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12, T = T + 1e-6*eye(M); end
        Tinv = inv(T);
        beta_AML = inv((A'*Tinv*A) .* (S*S').') * diag(A'*Tinv*X*S');
        mse_aml3 = mse_aml3 + norm(beta_AML(3) - beta_true(3))^2;
        
        % MAFI estimation from [28]
        % Estimate noise covariance matrix using residual from AML? In [28] they use sample covariance.
        % Here we use the sample covariance of X (since signal is weak at high SNR? but better to use T as an estimate of Q)
        % For MAFI, we need R = E[z z^H]. We can use T (which projects out signal) as an estimate.
        R_hat = T / (L - rank(S));  % unbiased estimate of Q
        if rcond(R_hat) < 1e-12, R_hat = R_hat + 1e-6*eye(M); end
        Rinv = inv(R_hat);
        % For each snapshot? MAFI in [28] uses the full data vector? Actually they use the whole data sequence.
        % Here we use the sum over snapshots: for each frequency, estimate amplitude by matched filter.
        % Define steering vector for each frequency (size M)
        a_n = exp(1j*2*pi*f(3)*(0:M-1)');   % for beta3
        % The data vector for each snapshot is column of X. Combine using average.
        % The ML estimator for amplitude with known R and known steering vector is:
        % beta_hat = (a^H R^{-1} a)^{-1} a^H R^{-1} x.
        % We apply to each snapshot and average.
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
    end
    MSE_AML_beta3(idx) = mse_aml3 / nMC;
    MSE_MAFI_beta3(idx) = mse_mafi3 / nMC;
    
    % CRB for beta3 (from [28, Eq.(9)])
    % CRB = (a^H R^{-1} a)^{-1} for each sinusoid, but with R = noise covariance.
    % For colored noise, R is Toeplitz with entries given by autocorrelation of AR(1) process.
    % Here we use true Q (since CRB uses true parameters).
    % Compute true Q for given sigma2_e.
    r0 = sigma2_e / (1 - ar_coef^2);   % noise variance
    r1 = ar_coef * r0;                 % lag-1 correlation
    Q_true = toeplitz(r0 * ones(1,M), [r0, r1*ones(1,M-1)]);
    % For complex noise, but AR(1) as defined is complex? Actually e(l) is complex white.
    % The AR process yields complex noise with same autocorrelation.
    % We'll use real autocorrelation for simplicity (but paper uses complex exponential correlation in array example).
    % For sinusoidal amplitude estimation, the CRB is given by (a^H Q^{-1} a)^{-1}.
    a3 = exp(1j*2*pi*f(3)*(0:M-1)');
    CRB_beta3(idx) = real(1 / (a3' * inv(Q_true) * a3));
    
    % === Subplot (b): beta1 ===
    sigma2_e = sigma2_e_beta1;
    mse_aml1 = 0; mse_mafi1 = 0;
    for mc = 1:nMC
        e = sqrt(sigma2_e/2) * (randn(1, L0) + 1j*randn(1, L0));
        z = filter(1, [1, -ar_coef], e);
        s_clean = zeros(1, L0);
        for n = 1:N
            s_clean = s_clean + beta_true(n) * exp(1j*2*pi*f(n)*(0:L0-1));
        end
        x = s_clean + z;
        X = hankel(x(1:M), x(M:end));
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12, T = T + 1e-6*eye(M); end
        Tinv = inv(T);
        beta_AML = inv((A'*Tinv*A) .* (S*S').') * diag(A'*Tinv*X*S');
        mse_aml1 = mse_aml1 + norm(beta_AML(1) - beta_true(1))^2;
        
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
        mse_mafi1 = mse_mafi1 + norm(beta_mafi(1) - beta_true(1))^2;
    end
    MSE_AML_beta1(idx) = mse_aml1 / nMC;
    MSE_MAFI_beta1(idx) = mse_mafi1 / nMC;
    
    a1 = exp(1j*2*pi*f(1)*(0:M-1)');
    r0 = sigma2_e / (1 - ar_coef^2);
    r1 = ar_coef * r0;
    Q_true = toeplitz(r0 * ones(1,M), [r0, r1*ones(1,M-1)]);
    CRB_beta1(idx) = real(1 / (a1' * inv(Q_true) * a1));
end

% Plot Fig.7
figure('Position', [100, 100, 1000, 400]);
subplot(1,2,1);
semilogy(local_SNR_dB_vals, MSE_AML_beta3, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(local_SNR_dB_vals, MSE_MAFI_beta3, 'bs-', 'LineWidth', 1.5);
semilogy(local_SNR_dB_vals, CRB_beta3, 'k--', 'LineWidth', 1.5);
xlabel('Local SNR (dB)'); ylabel('MSE');
legend('AML', 'MAFI', 'CRB', 'Location', 'best');
title('(a) For β_3');
grid on;

subplot(1,2,2);
semilogy(local_SNR_dB_vals, MSE_AML_beta1, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(local_SNR_dB_vals, MSE_MAFI_beta1, 'bs-', 'LineWidth', 1.5);
semilogy(local_SNR_dB_vals, CRB_beta1, 'k--', 'LineWidth', 1.5);
xlabel('Local SNR (dB)'); ylabel('MSE');
legend('AML', 'MAFI', 'CRB', 'Location', 'best');
title('(b) For β_1');
grid on;

sgtitle('Fig. 7: L_0 = 32, M = 8, colored AR(1) noise');