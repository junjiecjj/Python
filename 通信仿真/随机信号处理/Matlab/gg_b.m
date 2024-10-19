


% 7|随机信号分析与应用：从自相关到功率谱密度的探讨（中）
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485848&idx=1&sn=ae45c56025570a5faf28791eb8c71813&chksm=c15b98b6f62c11a0158a33ea3ba5d509a4d15cdb65d2a20504e789d89a665ae7a442f87e8167&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all


fs = 1000;
t = 0:1/fs:1-1/fs;
x = cos(2*pi*100*t) + 0.5*sin(2*pi*200*t) + randn(size(t));
segment_lengths = [64, 128, 256];  % 不同的分段长度
overlap_ratios = [0.25, 0.5, 0.75];  % 不同的重叠率
window_types = {'rectwin', 'hamming', 'hann', 'blackman'};  % 不同的窗口函数

%% 1. 分析分段长度的影响
figure;
for i = 1:length(segment_lengths)
    subplot(3, 1, i);
    [Pxx, f] = pwelch(x, segment_lengths(i), round(segment_lengths(i)*0.5), [], fs);
    plot(f, 10*log10(Pxx), 'LineWidth', 1.5);
    title(['分段长度 = ', num2str(segment_lengths(i))], 'FontSize', 12);
    xlabel('频率 (Hz)', 'FontSize', 12);
    ylabel('功率/频率 (dB/Hz)', 'FontSize', 12);
    grid on;
end

%% 2. 分析重叠率的影响
figure
for i = 1:length(overlap_ratios)
    subplot(3, 1, i);
    [Pxx, f] = pwelch(x, 128, round(128*overlap_ratios(i)), [], fs);
    plot(f, 10*log10(Pxx), 'LineWidth', 1.5);
    title(['重叠率 = ', num2str(overlap_ratios(i)*100), '%'], 'FontSize', 12);
    xlabel('频率 (Hz)', 'FontSize', 12);
    ylabel('功率/频率 (dB/Hz)', 'FontSize', 12);
    grid on;
end

figure
%% 3. 分析窗口函数的影响
for i = 1:length(window_types)
    subplot(2,2,i)
    window_function = window(str2func(window_types{i}), 128);
    [Pxx, f] = pwelch(x, window_function, 64, [], fs);
    plot(f, 10*log10(Pxx), 'LineWidth', 1.5);
    title([window_types{i}, ' 窗口'], 'FontSize', 12);
    xlabel('频率 (Hz)', 'FontSize', 12);
    ylabel('功率/频率 (dB/Hz)', 'FontSize', 12);
    grid on;
end

