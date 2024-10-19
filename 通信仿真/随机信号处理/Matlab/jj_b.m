


% 2 | 数据滤波：探讨卡尔曼滤波、SG滤波、组合滤波以及滑动平均滤波
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485991&idx=1&sn=e2a0834f9b54b58de2e9595e94da5283&chksm=c15b9b09f62c121fc31a365c26575b9c1cd61fb5b65e46b8a5e5953ff36aa702c7348b7a7736&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all


t = 0:0.01:10;

% 缓慢
slow_signal = sin(2*pi*0.5*t);

% 快速
fast_signal = sin(2*pi*5*t);
sigma = 0.5;
slow_signal_noisy = slow_signal + sigma*randn(size(t));
fast_signal_noisy = fast_signal + sigma*randn(size(t));
moving_avg = @(x, N) filter(ones(1,N)/N, 1, x);
window_sizes = [5, 10, 20, 50];

figure;
subplot(2,1,1);
plot(t, slow_signal, 'r-', t, slow_signal_noisy, 'b-');
title('缓慢变化信号与噪声信号');
legend('缓慢变化信号', '噪声信号');
subplot(2,1,2);
plot(t, fast_signal, 'r-', t, fast_signal_noisy, 'b-');
title('快速变化信号与噪声信号');
legend('快速变化信号', '噪声信号');

%% 1
for N = window_sizes
    % 滑动平均结果
    slow_filtered = moving_avg(slow_signal_noisy, N);
    fast_filtered = moving_avg(fast_signal_noisy, N);
    figure;
    subplot(2, 1, 1);
    plot(t, slow_signal, 'r-', t, slow_signal_noisy, 'b-', t, slow_filtered, 'g-');
    title(['缓慢变化信号: 窗口大小 = ', num2str(N)]);
    legend('原始信号', '噪声信号', '滑动平均信号');

    subplot(2, 1, 2);
    plot(t, fast_signal, 'r-', t, fast_signal_noisy, 'b-', t, fast_filtered, 'g-');
    title(['快速变化信号: 窗口大小 = ', num2str(N)]);
    legend('原始信号', '噪声信号', '滑动平均信号');
end

%% 2
figure;
hold on;
var_noise_slow = [];
var_noise_fast = [];
for N = window_sizes
    slow_filtered = moving_avg(slow_signal_noisy, N);
    fast_filtered = moving_avg(fast_signal_noisy, N);

    noise_slow = slow_filtered - slow_signal;
    noise_fast = fast_filtered - fast_signal;

    var_noise_slow = [var_noise_slow var(noise_slow)];
    var_noise_fast = [var_noise_fast var(noise_fast)];
end
plot(window_sizes, var_noise_slow, 'ro-', 'DisplayName', '缓慢变化信号的噪声方差');
plot(window_sizes, var_noise_fast, 'bo-', 'DisplayName', '快速变化信号的噪声方差');
xlabel('窗口大小');
ylabel('噪声方差');
title('不同窗口大小下的噪声方差');
legend;
hold off;


