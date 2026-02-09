
clear;
clc;
close all;
rng(42);

%global N Nrf K Pt;
% global Vrf;

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
num_iters = 100;

SNRdBs = -10:2:10;
RateLst = zeros(1,length(SNRdBs));
% parfor i = 1:length(SNRdBs) 
for i = 1:length(SNRdBs) 
    snrdB = SNRdBs(i);
    Pt = 10^(snrdB / 10) * sigma2;
    fprintf('main: snr idx = %d/%d \n', i, length(SNRdBs));
    for it = 1:num_iters
        fprintf('  repeat it = %d/%d \n', it, num_iters);
        H = channel(K, N, M, L);
        H = squeeze(H);
        tmp = Alg3(H, beta, Nrf, Pt, sigma2);
        RateLst(i) = RateLst(i) + tmp;
    end
    RateLst(i) = RateLst(i) / num_iters;
    fprintf('main: snr idx = %d/%d,  Cap = %.6f\n', i, length(SNRdBs), RateLst(i));
end



figure(1);
% 使用简写格式绘制
plot(SNRdBs, RateLst, 'b--o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Multi-User Large-Scale MISO');
xlabel('SNR(dB)', 'FontSize', 16);
ylabel('Spectral Efficiency(bits/s/Hz)', 'FontSize', 16);
grid on;
legend('FontSize', 16);
hold on;


