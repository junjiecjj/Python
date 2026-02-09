

clear;
clc;
close all;

rng(42,'twister');

% num of BS antennas
N = 64;
% num of receiver antennas
M = 16;
% num of users
K = 1;
% num of data streams per user
d = 6;
% num of data streams
Ns = K * d;
Nrf = Ns;
% stopping (convergence) condition
epsilon = 1e-4;
L = 15;

sigma2 = 40;
% num of iterations for each dB step
num_iters = 100;

SNRdBs = -10:2:6;
RateLst = zeros(1,length(SNRdBs));
parfor i = 1:length(SNRdBs)
    i
    snrdB = SNRdBs(i);
    P = 10^(snrdB / 10) * sigma2;
    for it = 1:num_iters
        H = channel(K, N, M, L);
        H = squeeze(H);
        tmp = Algo2(H, Ns, P, sigma2, epsilon);
        RateLst(i) = RateLst(i) + tmp;
    end
    RateLst(i) = RateLst(i) / num_iters;
end

figure(1);
% 使用简写格式绘制
plot(SNRdBs, RateLst, 'b--o', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'Single-User Large-Scale MIMO');
xlabel('SNR(dB)', 'FontSize', 16);
ylabel('Spectral Efficiency(bits/s/Hz)', 'FontSize', 16);
grid on;
legend('FontSize', 16);
hold on;



























