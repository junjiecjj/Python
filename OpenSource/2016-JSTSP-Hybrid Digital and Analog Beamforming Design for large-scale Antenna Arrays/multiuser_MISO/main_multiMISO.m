
clear;
clc;
close all;
rng(42,'twister');

%global N Nrf K Pt;
global Vrf;

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
num_iters = 10;

SNRdBs = -10:2:10;
RateLst = zeros(1,length(SNRdBs));
for i = 1:length(SNRdBs)
    i
    snrdB = SNRdBs(i);
    Pt = 10^(snrdB / 10) * sigma2;
    for it = 1:num_iters
        it
        H = channel(K, N, M, L);
        H = squeeze(H);
        tmp = Alg3(H, beta, Nrf, Pt, sigma2);
        RateLst(i) = RateLst(i) + tmp;
    end
    RateLst(i) = RateLst(i) / num_iters;
end





