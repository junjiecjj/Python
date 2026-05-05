% fig6.m: Fig. 6 in Xu, Stoica, Li 2006
% MSE vs SNR, L=128, N=13, M=6, linearly dependent steering vectors

clc;
clear all;
close all;

M = 6; N = 13; L = 128;
angles = linspace(-30, 30, N) * pi/180;
A = exp(1j * (0:M-1)' * (2*pi*0.5*sin(angles)));
S = (randn(N, L) + 1j*randn(N, L))/sqrt(2); % fixed across SNR
beta_true = ones(N, 1);
B = diag(beta_true);

SNR_dB_vals = -10:5:30;
nMC = 500;
MSE_AML = zeros(size(SNR_dB_vals));
MSE_LS = zeros(size(SNR_dB_vals));
CRB_vals = zeros(size(SNR_dB_vals));

for sidx = 1:length(SNR_dB_vals)
    SNR = 10^(SNR_dB_vals(sidx)/10);
    Q_true = 1/SNR * 0.9.^(abs((0:M-1)-(0:M-1)')) .* exp(1j*((0:M-1)-(0:M-1)')*pi/2);
    
    for mc = 1:nMC
        Z = (randn(M, L) + 1j*randn(M, L))/sqrt(2);
        noise = sqrtm(Q_true) * Z;
        X = A * B * S + noise;
        
        % AML
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12, T = T + 1e-6*eye(M); end
        Tinv = inv(T);
        beta_AML = inv((A'*Tinv*A) .* (S*S').') * diag(A'*Tinv*X*S');
        MSE_AML(sidx) = MSE_AML(sidx) + norm(beta_AML - beta_true)^2;
        
        % LS
        beta_LS = inv((A'*A) .* (S*S').') * diag(A'*X*S');
        MSE_LS(sidx) = MSE_LS(sidx) + norm(beta_LS - beta_true)^2;
    end
    MSE_AML(sidx) = MSE_AML(sidx)/nMC;
    MSE_LS(sidx) = MSE_LS(sidx)/nMC;
    
    CRB_vals(sidx) = trace(inv((A'*inv(Q_true)*A) .* (S*S').'));
end

figure;
semilogy(SNR_dB_vals, MSE_AML, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(SNR_dB_vals, MSE_LS, 'bs-', 'LineWidth', 1.5);
semilogy(SNR_dB_vals, CRB_vals, 'k--', 'LineWidth', 1.5);
xlabel('SNR (dB)'); ylabel('MSE');
legend('AML', 'LS', 'CRB', 'Location', 'best');
title('Fig. 6: L=128, M=6, N=13, linearly dependent steering vectors');
grid on;