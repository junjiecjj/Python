% fig5.m: Fig. 5 in Xu, Stoica, Li 2006
% MSE vs snapshot number L, SNR=10dB, N=13, M=6, linearly dependent steering vectors

clc;
clear all;
close all;

M = 6; 
N = 13; 
SNR_dB = 10; 
SNR = 10^(SNR_dB/10);
angles = linspace(-30, 30, N);
A = exp(1j * (0:M-1)' * (2*pi*0.5*sind(angles))); % steering vectors, rank <= M < N

L_vals = ceil(logspace(1, 3, 10));
nMC = 1000;
MSE_AML = zeros(size(L_vals));
MSE_LS = zeros(size(L_vals));
CRB_vals = zeros(size(L_vals));

% Generate independent waveforms (rows of S are linearly independent when L >= N)
% For each L, S is random (different MC runs will use different S, but CRB uses S)
% We generate S once per L for fair comparison, but CRB should use true Q (unknown in practice)
% For simplicity, we generate S per L and keep it fixed across MC runs.

for lidx = 1:length(L_vals)
    L = L_vals(lidx);
    S = (randn(N, L) + 1j*randn(N, L))/sqrt(2); % independent waveforms
    beta_true = ones(N, 1);
    B = diag(beta_true);
    
    for mc = 1:nMC
        Z = (randn(M, L) + 1j*randn(M, L))/sqrt(2);
        Q = 1/SNR * 0.9.^(abs((0:M-1)-(0:M-1)')) .* exp(1j*((0:M-1)-(0:M-1)')*pi/2);
        noise = sqrtm(Q) * Z;
        X = A * B * S + noise;
        
        % AML
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        % if rcond(T) < 1e-12
        %     T = T + 1e-6*eye(M); 
        % end
        beta_AML = pinv((A'/T*A) .* (S*S').') * diag(A'/T*X*S');
        MSE_AML(lidx) = MSE_AML(lidx) + norm(beta_AML(2) - beta_true(2))^2;
        
        % LS
        beta_LS = pinv((A'*A) .* (S*S').') * diag(A'*X*S');
        MSE_LS(lidx) = MSE_LS(lidx) + norm(beta_LS(2) - beta_true(2))^2;

    end
    MSE_AML(lidx) = MSE_AML(lidx)/nMC;
    MSE_LS(lidx) = MSE_LS(lidx)/nMC;
    % CRB
    tmp = diag(inv((A'*pinv(Q)*A) .* (S*S').'));
    CRB_vals(lidx) = tmp(2);
end

figure(1);
loglog(L_vals, MSE_AML, 'ro-', 'LineWidth', 1.5); hold on;
loglog(L_vals, MSE_LS, 'bs-', 'LineWidth', 1.5);
loglog(L_vals, CRB_vals, 'k--', 'LineWidth', 1.5);
xlabel('Snapshot number L'); ylabel('MSE');
legend('AML', 'LS', 'CRB', 'Location', 'best');
title('Fig. 5: SNR=10dB, M=6, N=13, linearly dependent steering vectors');
grid on;