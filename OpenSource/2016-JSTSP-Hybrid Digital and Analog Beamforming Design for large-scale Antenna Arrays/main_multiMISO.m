
clear;
clc;
close all;
rng(42,'twister');

% num of BS antennas
N = 64;
% num of receiver antennas
M = 1;
% num of users
K = 8;
% num of data streams per user
d = 1;
% num of data streams
Ns = K * d;
Nrf = 9;
% stopping (convergence) condition
epsilon = 1e-4;
L = 15;
% 均匀分配权重
beta = ones(1,K);
sigma2 = K;
% num of iterations for each dB step
num_iters = 1000;

SNRdBs = -10:2:6;
RateLst = zeros(1,length(SNRdBs));
for i = 1:length(SNRdBs)
    i
    snrdB = SNRdBs(i);
    P = 10^(snrdB / 10) * sigma2;
    for it = 1:num_iters
        H = channel(K, N, M, L);
        H = squeeze(H);
        tmp = Alg3(H, beta, Nrf, Ps, sigma2);
        RateLst(i) = RateLst(i) + tmp;
    end
    RateLst(i) = RateLst(i) / num_iters;
end

function a = stevec_ULA(theta, M)
    % Generates a steering vector for Uniform Linear Array (ULA)
    % theta: rad
    m = 0:M-1;
    a = exp(1i * pi * m * sind(theta))/sqrt(M);
    a = a.';
end

function H = channel(K, N, M, L)
    H = zeros(K, M, N);
    for k = 1:K
        phi_t = 2*pi*rand(L);
        phi_r = 2*pi*rand(L);
        alphas = randn(L);
        Hk = zeros(M, N);
        for l = 1:L 
            at = stevec_ULA(phi_t(l), N);
            ar = stevec_ULA(phi_r(l), M);
            Hk = Hk + alphas(l) * (ar * at');
        end
        H(k,:,:) = Hk;
    end
    H = H * sqrt(N*M/L); 
end


function rate = Alg3(H, beta, Nrf, P, sigma2)
    rate = 0;

end 
