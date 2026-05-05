% fig1.m: Fig. 1 in Xu, Stoica, Li 2006
% MSE vs snapshot number L, SNR=10dB, independent waveforms

clc;
clear all;
close all;

M = 6; 
N = 3; 
SNR_dB = 10; 
SNR = 10^(SNR_dB/10);
angles = [-10, 5, 10] * pi/180;
A = exp(1j * (0:M-1)' * (2*pi*0.5*sin(angles))); % steering vectors

L_vals = 20:20:200;
nMC = 1000;
MSE_AML = zeros(size(L_vals));
MSE_LS = zeros(size(L_vals));
MSE_GC = zeros(size(L_vals));
CRB_vals = zeros(size(L_vals));

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
        if rcond(T) < 1e-12
            T = T + 1e-6*eye(M); 
        end
        Tinv = inv(T);
        beta_AML = pinv((A'/T*A) .* (S*S').') * diag(A'/T*X*S');
        MSE_AML(lidx) = MSE_AML(lidx) + norm(beta_AML - beta_true)^2;
        
        % LS
        beta_LS = pinv((A'*A) .* (S*S').') * diag(A'*X*S');
        MSE_LS(lidx) = MSE_LS(lidx) + norm(beta_LS - beta_true)^2;
        
        % GC (unconstrained)
        B_GC = pinv(A) * X * pinv(S);
        MSE_GC(lidx) = MSE_GC(lidx) + norm(diag(B_GC) - beta_true)^2;
    end
    MSE_AML(lidx) = MSE_AML(lidx)/nMC;
    MSE_LS(lidx) = MSE_LS(lidx)/nMC;
    MSE_GC(lidx) = MSE_GC(lidx)/nMC;
    
    % CRB
    CRB_vals(lidx) = trace(inv((A'*pinv(Q)*A) .* (S*S').'));
end

figure;
semilogy(L_vals, MSE_AML, 'ro-', 'LineWidth', 1.5); hold on;
semilogy(L_vals, MSE_LS, 'bs-', 'LineWidth', 1.5);
semilogy(L_vals, MSE_GC, 'gd-', 'LineWidth', 1.5);
semilogy(L_vals, CRB_vals, 'k--', 'LineWidth', 1.5);
xlabel('Snapshot number L'); ylabel('MSE');
legend('AML', 'LS', 'GC', 'CRB', 'Location', 'best');
title('Fig. 1: SNR=10dB, M=6, N=3, independent waveforms');
grid on;


function C = khatri_rao(A, B)
% KHATRI_RAO  Khatri–Rao product (column-wise Kronecker product)
%   C = khatri_rao(A, B) returns a matrix C of size (size(A,1)*size(B,1)) x size(A,2)
%   such that C(:,i) = kron(A(:,i), B(:,i)).
%
%   Inputs:
%       A - m x p matrix
%       B - n x p matrix
%   Output:
%       C - (m*n) x p matrix
%
%   Example:
%       A = [1 2; 3 4]; B = [5 6; 7 8];
%       C = khatri_rao(A, B); % size 4x2
%       % C(:,1) = kron(A(:,1), B(:,1)) = [1*5; 1*7; 3*5; 3*7] = [5;7;15;21]
%       % C(:,2) = kron(A(:,2), B(:,2)) = [2*6; 2*8; 4*6; 4*8] = [12;16;24;32]

    if size(A,2) ~= size(B,2)
        error('A and B must have the same number of columns.');
    end
    p = size(A,2);
    m = size(A,1);
    n = size(B,1);
    C = zeros(m*n, p);
    for i = 1:p
        C(:,i) = kron(A(:,i), B(:,i));
    end
end