% fig4.m: Fig.4 in Xu, Stoica, Li 2006 (corrected)
% MSE vs SNR, L=128, M=6, N=3, independent waveforms
% Only show MSE for the signal arriving from theta2 = 5 deg.

clc;
clear all;
close all;


M = 6; 
N = 3; 
L = 128;
angles = [-10, 5, 10];
A = exp(1j * (0:M-1)' * (2*pi*0.5*sind(angles)));

beta_true = ones(N, 1);
B = diag(beta_true);

SNR_dB_vals = -20:5:20;
nMC = 500;

MSE_AML = zeros(size(SNR_dB_vals));
MSE_LS  = zeros(size(SNR_dB_vals));
MSE_GC  = zeros(size(SNR_dB_vals));
CRB_vals = zeros(size(SNR_dB_vals));

for sidx = 1:length(SNR_dB_vals)
    SNR = 10^(SNR_dB_vals(sidx)/10);
    Q_true = 1/SNR * 0.9.^(abs((0:M-1)-(0:M-1)')) .* exp(1j*((0:M-1)-(0:M-1)')*pi/2);
    mse_aml = 0; 
    mse_ls = 0; 
    mse_gc = 0;
    for mc = 1:nMC
        % 相同波形：所有行相同（随机生成一个行向量，再复制 N 行）
        s = (randn(1, L) + 1j*randn(1, L))/sqrt(2);
        S = repmat(s, N, 1);
        Z = (randn(M, L) + 1j*randn(M, L))/sqrt(2);
        noise = sqrtm(Q_true) * Z;
        X = A * B * S + noise;
        
        % AML
        Pi_S = S' * pinv(S*S') * S;
        T = X * (eye(L) - Pi_S) * X';
        if rcond(T) < 1e-12
            T = T + 1e-6*eye(M); 
        end
        beta_AML = pinv((A'/T*A) .* (S*S').') * diag(A'/T*X*S');
        mse_aml = mse_aml + abs(beta_AML(2) - beta_true(2))^2;
        
        % LS
        beta_LS = pinv((A'*A) .* (S*S').') * diag(A'*X*S');
        mse_ls = mse_ls + abs(beta_LS(2) - beta_true(2))^2;
        
        % GC
        B_GC = pinv(A) * X * pinv(S);
        mse_gc = mse_gc + abs(B_GC(2,2) - beta_true(2))^2;
    end
    MSE_AML(sidx) = mse_aml / nMC;
    MSE_LS(sidx)  = mse_ls / nMC;
    MSE_GC(sidx)  = mse_gc / nMC;
    
    CRB_matrix = inv((A'*pinv(Q_true)*A) .* (S*S').');
    CRB_vals(sidx) = real(CRB_matrix(2,2));
end

figure;
semilogy(SNR_dB_vals, MSE_AML, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(SNR_dB_vals, MSE_LS,  'bs-', 'LineWidth', 1.5);
semilogy(SNR_dB_vals, MSE_GC,  'gd-', 'LineWidth', 1.5);
semilogy(SNR_dB_vals, CRB_vals, 'k--', 'LineWidth', 1.5);
xlabel('SNR (dB)');
ylabel('MSE');
legend('AML', 'LS', 'GC (unconstrained)', 'CRB', 'Location', 'best');
title('Fig. 2: L=128, M=6, N=3, independent waveforms, MSE for θ₂=5°');
grid on;