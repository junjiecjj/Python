% fig3.m: Fig. 3 in Xu, Stoica, Li 2006
% MSE vs snapshot number L, identical waveforms

clc;
clear all;
close all;

M = 6; N = 3; SNR_dB = 10; SNR = 10^(SNR_dB/10);
angles = [-10, 5, 10] * pi/180;
A = exp(1j * (0:M-1)' * (2*pi*0.5*sin(angles)));

L_vals = 20:20:200;
nMC = 500;
MSE_AML = zeros(size(L_vals));
MSE_LS = zeros(size(L_vals));
CRB_vals = zeros(size(L_vals));

for lidx = 1:length(L_vals)
    L = L_vals(lidx);
    s = (randn(1, L) + 1j*randn(1, L))/sqrt(2);
    S = repmat(s, N, 1); % identical waveforms
    beta_true = ones(N, 1);
    B = diag(beta_true);
    
    for mc = 1:nMC
        Z = (randn(M, L) + 1j*randn(M, L))/sqrt(2);
        Q = 1/SNR * 0.9.^(abs((0:M-1)-(0:M-1)')) .* exp(1j*((0:M-1)-(0:M-1)')*pi/2);
        noise = sqrtm(Q) * Z;
        X = A * B * S + noise;
        
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12, T = T + 1e-6*eye(M); end
        Tinv = inv(T);
        beta_AML = inv((A'*Tinv*A) .* (S*S').') * diag(A'*Tinv*X*S');
        MSE_AML(lidx) = MSE_AML(lidx) + norm(beta_AML - beta_true)^2;
        
        beta_LS = inv((A'*A) .* (S*S').') * diag(A'*X*S');
        MSE_LS(lidx) = MSE_LS(lidx) + norm(beta_LS - beta_true)^2;
    end
    MSE_AML(lidx) = MSE_AML(lidx)/nMC;
    MSE_LS(lidx) = MSE_LS(lidx)/nMC;
    CRB_vals(lidx) = trace(inv((A'*inv(Q)*A) .* (S*S').'));
end

figure;
semilogy(L_vals, MSE_AML, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(L_vals, MSE_LS, 'bs-', 'LineWidth', 1.5);
semilogy(L_vals, CRB_vals, 'k--', 'LineWidth', 1.5);
xlabel('Snapshot number L'); ylabel('MSE');
legend('AML', 'LS', 'CRB', 'Location', 'best');
title('Fig. 3: SNR=10dB, M=6, N=3, identical waveforms');
grid on;