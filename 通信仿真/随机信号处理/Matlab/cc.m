% 3 | 随机信号分析与应用：从自相关到功率谱密度的探讨
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485721&idx=1&sn=d410026abf27fd73d43f00aeef244e88&chksm=c15b9837f62c1121c868566b0be9eadb564edefbc0a8c360e95432fb8873ab13299adfbf823b&cur_album_id=3587607448191893505&scene=190#rd

close all;
clear all;
clc;




clc
clear
close all

fs = 1000;  % 采样频率
t = 0:1/fs:1-1/fs;  

% signal = sin(2*pi*50*t) + sin(2*pi*120*t) + randn(size(t));
% signal = sin(2*pi*50*t);
% signal = randn(1, size(t,2));
signal = sin(2*pi*50*t) + sin(2*pi*120*t);

spectral_analysis_comparison(signal, fs);


function spectral_analysis_comparison(signal, fs)
    % signal: 输入信号
    % fs: 采样频率

    N = length(signal);  % 信号长度
    L = 256;             % Welch方法中的子段长度
    D = L/2;             % 重叠长度

    [Pxx_periodogram, ~] = periodogram_method(signal, fs, N);
    [Pxx_correlogram, f] = correlogram_method(signal, fs, N);
    [Pxx_welch, f_welch] = welch_method(signal, fs, L, D);
    Pxx_welch = interp1(f_welch, Pxx_welch, f, 'linear', 'extrap');%

    figure;
    subplot(3,1,1);
    plot(f, 10*log10(Pxx_periodogram), 'LineWidth', 1.5); hold on;
    plot(f, 10*log10(Pxx_correlogram), 'LineWidth', 1.5);
    plot(f, 10*log10(Pxx_welch), 'LineWidth', 1.5);
    xlabel('频率 (Hz)');
    ylabel('功率/频率 (dB/Hz)');
    title('功率谱密度估计');
    legend('周期图法', '自相关函数法', 'Welch 方法');
    grid on;
    subplot(3,1,2);
    var_periodogram = var(Pxx_periodogram);
    var_correlogram = var(Pxx_correlogram);
    var_welch = var(Pxx_welch);
    bar([1 2 3], [var_periodogram, var_correlogram, var_welch]);
    set(gca, 'XTickLabel', {'周期图法', '自相关函数法', 'Welch 方法'});
    ylabel('估计方差');
    title('不同方法的估计方差比较');
    grid on;

    subplot(3,1,3);
    res_periodogram = diff(f);
    res_welch = diff(f_welch);
    bar(1:2, [mean(res_periodogram), mean(res_welch)]);
    set(gca, 'XTickLabel', {'周期图法 & 自相关函数法', 'Welch 方法'});
    ylabel('频谱分辨率 (Hz)');
    title('不同方法的频谱分辨率比较');
    grid on;
end

function [Pxx, f] = periodogram_method(signal, fs, N)
    % 计算周期图法的功率谱密度
    X = fft(signal, N);
    Pxx = (1/N) * abs(X).^2;
    Pxx = Pxx(1:N/2+1);  % 取前半部分（正频率）
    f = (0:N/2) * (fs/N);
end

function [Pxx, f] = correlogram_method(signal, fs, N)
    % 计算基于自相关函数法的功率谱密度
    r = xcorr(signal, 'biased');
    r = r(N:end);  % 取正半轴部分
    Rxx = fft(r, N);
    Pxx = abs(Rxx(1:N/2+1));  % 取前半部分（正频率）
    f = (0:N/2) * (fs/N);
end

function [Pxx, f] = welch_method(signal, fs, L, D)
    % 计算Welch方法的功率谱密度
    [Pxx, f] = pwelch(signal, hamming(L), D, L, fs);
end
