% 4 | 随机信号分析与应用：从自相关到功率谱密度的探讨
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485778&idx=1&sn=09ec65d7f65cc25cb944f0b895b6a05b&chksm=c15b987cf62c116a7cccf99e57ed8314ca03f77b6763b48a1152eae5aeeb7dd8e1583c049a0c&cur_album_id=3587607448191893505&scene=190#rd


close all;
clear all;
clc;

% 场景 1: 通信领域的信号分析
% 参数设置
Fs = 10000;              % 采样频率 (Hz)
T = 1;                   % 信号持续时间 (秒)
t = 0:1/Fs:T-1/Fs;       % 时间向量
f_signal = 2000;         % 通信信号频率 (Hz)
A_signal = 1;            % 信号幅度

signal = A_signal * cos(2*pi*f_signal*t);  % 通信信号
interference = 0.3 * cos(2*pi*3000*t);     % 干扰信号 (3000 Hz)
noise = 0.1 * randn(size(t));              % 白噪声
x = signal + interference + noise;         % 最终信号

nfft = 2048;
[pxx, f] = pwelch(x, [], [], nfft, Fs);

figure;
subplot(2, 1, 1);
plot(t, x);
title('含干扰的通信信号');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2, 1, 2);
plot(f, 10*log10(pxx));
title('通信信号的功率谱密度 (dB/Hz)');
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');

% 场景 2: 医学图像处理中的噪声抑制
% 参数设置
Fs = 1000;               % 采样频率 (Hz)
T = 1;                   % 信号持续时间 (秒)
t = 0:1/Fs:T-1/Fs;       % 时间向量
f_signal = 50;           % 医学信号频率 (Hz)
A_signal = 1;            % 信号幅度

% 生成含高频噪声的医学信号
signal = A_signal * sin(2*pi*f_signal*t);  % 医学信号
noise = 0.5 * sin(2*pi*300*t) + 0.1 * randn(size(t)); % 高频噪声和白噪声
x = signal + noise;                        % 最终信号
nfft = 1024;
[pxx, f] = pwelch(x, [], [], nfft, Fs);

figure;
subplot(2, 1, 1);
plot(t, x);
title('含高频噪声的医学信号');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2, 1, 2);
plot(f, 10*log10(pxx));
title('医学信号的功率谱密度 (dB/Hz)');
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');

% 场景 3: 地震波分析


% 参数设置
Fs = 100;                % 采样频率 (Hz)
T = 10;                  % 信号持续时间 (秒)
t = 0:1/Fs:T-1/Fs;       % 时间向量

% 模拟地震波信号
f1 = 2; f2 = 5;          % 主要地震波频率 (Hz)
signal = cos(2*pi*f1*t) + 0.5*cos(2*pi*f2*t); % 地震波
noise = 0.2 * randn(size(t));                % 白噪声
x = signal + noise;                          % 最终信号
nfft = 2048;
[pxx, f] = pwelch(x, [], [], nfft, Fs);

figure;
subplot(2, 1, 1);
plot(t, x);
title('模拟地震波信号');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2, 1, 2);
plot(f, 10*log10(pxx));
title('地震波信号的功率谱密度 (dB/Hz)');
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');


% 场景 4: 音频处理和降噪


% 参数设置
Fs = 8000;               % 采样频率 (Hz)
T = 2;                   % 信号持续时间 (秒)
t = 0:1/Fs:T-1/Fs;       % 时间向量
f_signal = 1000;         % 音频信号频率 (Hz)
A_signal = 1;            % 信号幅度

% 生成含背景噪声的音频信号
signal = A_signal * cos(2*pi*f_signal*t);  % 音频信号
noise = 0.3 * randn(size(t));              % 白噪声
x = signal + noise;                        % 最终信号
nfft = 2048;
[pxx, f] = pwelch(x, [], [], nfft, Fs);

figure;
subplot(2, 1, 1);
plot(t, x);
title('含背景噪声的音频信号');
xlabel('时间 (s)');
ylabel('幅度');

subplot(2, 1, 2);
plot(f, 10*log10(pxx));
title('音频信号的功率谱密度 (dB/Hz)');
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');


% 4.3 功率谱密度的计算方法

clc;
clear;
close all

% 参数设置
Fs = 1000;       % 采样频率 (Hz)
T = 1/Fs;        % 采样周期 (s)
L = 1000;        % 信号长度
t = (0:L-1)*T;   % 时间向量

%% 生成测试信号
% 1. 纯正弦波信号
f1 = 50;        % 频率为50Hz的正弦波
x1 = sin(2*pi*f1*t);

% 2. 带噪声的正弦波信号
x2 = x1 + 0.5*randn(size(t));

% 3. 随机信号
x3 = randn(size(t));

%% 求解方法
methods = {'DFT', '自相关函数'};
for method = methods
    method = method{1};

    figure('Name', ['使用 ', method, ' 方法估计功率谱密度']);

    for i = 1:3
        x = eval(['x', num2str(i)]);

        if strcmp(method, 'DFT')
            % 直接傅里叶变换法
            X = fft(x);
            Pxx = (1/L) * abs(X).^2;
            f = Fs*(0:(L/2))/L;
            Pxx = Pxx(1:L/2+1);
        else
            % 基于自相关函数的方法
            Rxx = xcorr(x, 'biased');
            Rxx = Rxx(L:end);
            Pxx = abs(fft(Rxx, L));
            f = Fs*(0:(L/2))/L;
            Pxx = Pxx(1:L/2+1);
        end

        subplot(3, 1, i);
        plot(f, 10*log10(Pxx), 'LineWidth', 1.5);
        grid on;
        title(['信号 ', num2str(i)], 'FontSize', 12);
        xlabel('频率 (Hz)', 'FontSize', 12);
        ylabel('功率/频率 (dB/Hz)', 'FontSize', 12);
    end
end












