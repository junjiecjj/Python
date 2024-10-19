


% 9 | 随机信号分析与应用：从自相关到功率谱密度的探讨（中）
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485945&idx=1&sn=892f61b56b38aa214a1d6cfe2d708698&chksm=c15b98d7f62c11c1ad60655d7cd9a6f83174396b1dd25ab8fee08016fa0a7ab4f297a9455c69&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all


fs = 1000;              % 采样频率 (Hz)
t = 0:1/fs:10-1/fs;     % 时间向量 (10秒)
f1 = 50;                % 信号1的频率 (Hz)
f2 = 150;               % 信号2的频率 (Hz)
x = sin(2*pi*f1*t);
y = cos(2*pi*f2*t);
noise_level = 0.9;
x_noisy = x + noise_level * randn(size(t));
y_noisy = y + noise_level * randn(size(t));
nfft = 1024;
window = hamming(512);   %
noverlap = 256;

[Sxy_direct, f_direct] = cpsd(x_noisy, y_noisy, window, noverlap, nfft, fs);
x_noisy_detrend = detrend(x_noisy);
y_noisy_detrend = detrend(y_noisy);
[c, lags] = xcorr(x_noisy_detrend, y_noisy_detrend, 'biased');
Rxy = fft(c, nfft);
f_xcorr = (0:nfft-1)*(fs/nfft);
Rxy = Rxy / length(x_noisy_detrend);

%%
figure;
subplot(2,1,1);
plot(f_direct, abs(Sxy_direct));
title('直接傅里叶变换法估计的互谱密度');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0 200]);
grid on;

subplot(2,1,2);
plot(f_xcorr, abs(Rxy));
title('基于互相关函数法估计的互谱密度');
xlabel('频率 (Hz)');
ylabel('幅度');
xlim([0 200]);
grid on;
