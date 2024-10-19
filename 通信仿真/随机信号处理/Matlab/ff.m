% 6 | 随机信号分析与应用：从自相关到功率谱密度的探讨
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485820&idx=1&sn=a4eb803a233a3368ef601285cdc7072e&chksm=c15b9852f62c1144b7ee6c135242675c927fbc9831f9766e3a8c8ba14b776899030f628dfbfa&cur_album_id=3587607448191893505&scene=190&poc_token=HEAdCWej7OFMYwRATXjtKhAtv1L2buZdrdVqpuJU

% 4. 频域与时域关系的应用
% 例1：滤波器设计

    clc
    clear
    close all

    fs = 1000;                   % 采样频率 (Hz)
    t = 0:1/fs:1-1/fs;           % 时间向量
    f1 = 50;                     % 信号1的频率 (Hz)
    f2 = 200;                    % 信号2的频率 (Hz)

    x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t);
    noise = 0.3*randn(size(t));
    x_noisy = x + noise;
    cutoff = 100;
    [b, a] = butter(4, cutoff/(fs/2), 'low');
    x_filtered = filter(b, a, x_noisy);

    n = length(t);
    X_noisy = fft(x_noisy, n);
    X_filtered = fft(x_filtered, n);
    f = (0:n-1)*(fs/n);

    figure;
    subplot(3,1,1);
    plot(t, x_noisy);
    title('含噪信号 (时域)');
    xlabel('时间 (s)');
    ylabel('幅度');
    subplot(3,1,2);
    plot(t, x_filtered);
    title('滤波后的信号 (时域)');
    xlabel('时间 (s)');
    ylabel('幅度');
    subplot(3,1,3);
    plot(f, abs(X_noisy), 'r');
    hold on;
    plot(f, abs(X_filtered), 'b');
    title('滤波前后的信号 (频域)');
    xlabel('频率 (Hz)');
    ylabel('幅度');
    legend('滤波前', '滤波后');

% 例2：信号调制与解调
clc
clear
close all

fs = 1000;                   % 采样频率 (Hz)
t = 0:1/fs:1-1/fs;           % 时间向量
%     f1 = 50;                     % 信号1的频率 (Hz)
%     f2 = 200;                    % 信号2的频率 (Hz)

fc = 100;                   % 载波频率 (Hz)
fm = 10;                    % 调制信号频率 (Hz)
Am = 1;                     % 调制信号幅度
Ac = 1;                     % 载波信号幅度

m = Am * cos(2*pi*fm*t);
c = Ac * cos(2*pi*fc*t);
s = (1 + m) .* c;
figure;
subplot(3,1,1);
plot(t, m);
title('调制信号 (时域)');
xlabel('时间 (s)');
ylabel('幅度');
subplot(3,1,2);
plot(t, c);
title('载波信号 (时域)');
xlabel('时间 (s)');
ylabel('幅度');

subplot(3,1,3);
plot(t, s);
title('调制信号 (AM, 时域)');
xlabel('时间 (s)');
ylabel('幅度');

s_demod = abs(hilbert(s));  % 解调信号
figure;
plot(t, s_demod);
title('解调信号 (时域)');
xlabel('时间 (s)');
ylabel('幅度');


% 例3：噪声分析与消除
clc
clear
close  all
fs = 1000;                   % 采样频率 (Hz)
t = 0:1/fs:1-1/fs;           % 时间向量
%     f1 = 50;                     % 信号1的频率 (Hz)
%     f2 = 200;                    % 信号2的频率 (Hz)

f_noise = 50;

x_signal = sin(2*pi*10*t);
x_noise = 0.3 * sin(2*pi*f_noise*t);
x_total = x_signal + x_noise;

[b, a] = butter(2, [(f_noise-2)/(fs/2) (f_noise+2)/(fs/2)], 'stop');
x_denoised = filter(b, a, x_total);
[pxx_total, f_pxx] = periodogram(x_total, [], [], fs);
[pxx_denoised, ~] = periodogram(x_denoised, [], [], fs);

figure;
subplot(3,1,1);
plot(t, x_total);
title('含50Hz噪声的信号 (时域)');
xlabel('时间 (s)');
ylabel('幅度');

subplot(3,1,2);
plot(t, x_denoised);
title('滤波后的信号 (时域)');
xlabel('时间 (s)');
ylabel('幅度');

subplot(3,1,3);
plot(f_pxx, 10*log10(pxx_total), 'r');
hold on;
plot(f_pxx, 10*log10(pxx_denoised), 'b');
title('滤波前后的信号 (功率谱密度)');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');
legend('滤波前', '滤波后');

% 6.3 实例分析：如何通过自相关函数获得功率谱密度
clc
clear
close all

A = 1;           % 正弦波幅度
f0 = 5;          % 正弦波频率 (Hz)
Fs = 100;        % 采样频率 (Hz)
T = 1;           % 信号持续时间 (s)
sigma = 0.5;     % 噪声标准差
t = 0:1/Fs:T-1/Fs;
x = A * cos(2*pi*f0*t) + sigma * randn(size(t));
[R_x, lags] = xcorr(x, 'biased');
f = (-Fs/2:Fs/length(R_x):Fs/2-Fs/length(R_x));
S_x = abs(fftshift(fft(R_x)));

figure;
subplot(2,1,1);
plot(lags/Fs, R_x, 'LineWidth', 1.5);
grid on;
title('自相关函数 R_x(\tau)', 'FontSize', 14);
xlabel('时间延迟 \tau (秒)', 'FontSize', 12);
ylabel('自相关函数 R_x(\tau)', 'FontSize', 12);
subplot(2,1,2);
plot(f, S_x, 'LineWidth', 1.5);
grid on;
title('功率谱密度 S_x(f)', 'FontSize', 14);
xlabel('频率 (Hz)', 'FontSize', 12);
ylabel('功率谱密度 S_x(f)', 'FontSize', 12);












