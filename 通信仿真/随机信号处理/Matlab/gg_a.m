


% 7 | 随机信号分析与应用：从自相关到功率谱密度的探讨（上）
% https://mp.weixin.qq.com/s?__biz=MzkxNTcyMDI1Nw==&mid=2247485847&idx=1&sn=f7f0be6bdc5998df37baacb05d20311b&chksm=c15b98b9f62c11af04b39092d5ddd3b7e3231724a7d9ccc619cff95594f5fd84b4890916d0b5&cur_album_id=3587607448191893505&scene=190#rd


clc
clear
close all

fs = 1000;           % 采样频率
t = 0:1/fs:1-1/fs;   % 时间序列
f1 = 50;             % 第一个频率分量
f2 = 150;            % 第二个频率分量
f3 = 300;            % 第三个频率分量

x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.2*sin(2*pi*f3*t);%

x = x - mean(x);
hamming_window = hamming(length(x))';
rect_window = rectwin(length(x))';
blackman_window = blackman(length(x))';

x_hamming = x .* hamming_window;
x_rect = x .* rect_window;
x_blackman = x .* blackman_window;

[R_hamming_biased, lag] = xcorr(x_hamming, 'biased');
[R_hamming_unbiased, ~] = xcorr(x_hamming, 'unbiased');
[R_rect_biased, ~] = xcorr(x_rect, 'biased');
[R_blackman_biased, ~] = xcorr(x_blackman, 'biased');

S_hamming_biased = fftshift(fft(R_hamming_biased));
S_hamming_unbiased = fftshift(fft(R_hamming_unbiased));
S_rect_biased = fftshift(fft(R_rect_biased));
S_blackman_biased = fftshift(fft(R_blackman_biased));
f = (-length(S_hamming_biased)/2:length(S_hamming_biased)/2-1)*(fs/length(S_hamming_biased));
%%
figure;
subplot(4,3,1);
plot(t, x, 'b');
xlabel('时间 (s)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('去均值后的信号', 'FontSize', 12);
grid on;

subplot(4,3,2);
plot(t, x_hamming, 'r');
xlabel('时间 (s)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('Hamming窗', 'FontSize', 12);
grid on;

subplot(4,3,3);
plot(t, x_rect, 'g');
xlabel('时间 (s)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('矩形窗', 'FontSize', 12);
grid on;

subplot(4,3,4);
plot(t, x_blackman, 'm');
xlabel('时间 (s)', 'FontSize', 10);
ylabel('幅值', 'FontSize', 10);
title('Blackman窗', 'FontSize', 12);
grid on;

subplot(4,3,5);
plot(lag/fs, R_hamming_biased, 'k');
xlabel('时间滞后 (s)', 'FontSize', 10);
ylabel('自相关值', 'FontSize', 10);
title('Hamming窗下的有偏自相关', 'FontSize', 12);
grid on;

subplot(4,3,6);
plot(lag/fs, R_hamming_unbiased, 'c');
xlabel('时间滞后 (s)', 'FontSize', 10);
ylabel('自相关值', 'FontSize', 10);
title('Hamming窗下的无偏自相关', 'FontSize', 12);
grid on;

subplot(4,3,7);
plot(lag/fs, R_rect_biased, 'y');
xlabel('时间滞后 (s)', 'FontSize', 10);
ylabel('自相关值', 'FontSize', 10);
title('矩形窗下的有偏自相关', 'FontSize', 12);
grid on;

subplot(4,3,8);
plot(lag/fs, R_blackman_biased, 'b');
xlabel('时间滞后 (s)', 'FontSize', 10);
ylabel('自相关值', 'FontSize', 10);
title('Blackman窗下的有偏自相关', 'FontSize', 12);
grid on;

subplot(4,3,9);
plot(f, abs(S_hamming_biased), 'r');
xlabel('频率 (Hz)', 'FontSize', 10);
ylabel('功率谱密度', 'FontSize', 10);
title('Hamming窗下的有偏功率谱', 'FontSize', 12);
grid on;
xlim([0 fs/2]);

subplot(4,3,10);
plot(f, abs(S_hamming_unbiased), 'c');
xlabel('频率 (Hz)', 'FontSize', 10);
ylabel('功率谱密度', 'FontSize', 10);
title('Hamming窗下的无偏功率谱', 'FontSize', 12);
grid on;
xlim([0 fs/2]);

subplot(4,3,11);
plot(f, abs(S_rect_biased), 'g');
xlabel('频率 (Hz)', 'FontSize', 10);
ylabel('功率谱密度', 'FontSize', 10);
title('矩形窗下的有偏功率谱', 'FontSize', 12);
grid on;
xlim([0 fs/2]);

subplot(4,3,12);
plot(f, abs(S_blackman_biased), 'm');
xlabel('频率 (Hz)', 'FontSize', 10);
ylabel('功率谱密度', 'FontSize', 10);
title('Blackman窗下的有偏功率谱', 'FontSize', 12);
grid on;
xlim([0 fs/2]);


% 7.2 快速傅里叶变换（FFT）法

clc
clear
close all

fs = 1000;           % 采样频率 (Hz)
t = 0:1/fs:1-1/fs;   % 时间序列
f1 = 50;             % 第一个频率分量 (Hz)
f2 = 150;            % 第二个频率分量 (Hz)
f3 = 300;            % 第三个频率分量 (Hz)

x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.2*sin(2*pi*f3*t);
x = x - mean(x);
window = hanning(length(x))'; % 使用 Hanning 窗

figure;
% -------- 信号长度的影响 --------
% 选择不同的信号长度 (确保信号长度不超过原始信号长度)
Ns = [256, 512, 1000];  % 信号长度，不超过1000点
for i = 1:length(Ns)
    x_segment = x(1:Ns(i)); % 提取不同长度的信号段
    x_windowed = x_segment .* window(1:Ns(i));

    % 进行 FFT 计算
    N = 2^nextpow2(length(x_windowed));
    X = fft(x_windowed, N);
    X_magnitude = abs(X/N);
    f = (0:N/2-1)*(fs/N);

    % 计算功率谱密度 (PSD)
    Pxx = (1/(fs*N)) * abs(X(1:N/2)).^2;

    % 绘制不同信号长度的 PSD
    subplot(3, 3, i);
    plot(f, 10*log10(Pxx), 'b');
    xlabel('频率 (Hz)', 'FontSize', 10);
    ylabel('功率谱密度 (dB/Hz)', 'FontSize', 10);
    title(['信号长度 ' num2str(Ns(i))], 'FontSize', 12);
    grid on;
end

% -------- 零填充的影响 --------
x_windowed = x .* window;  % 使用完整信号，应用窗口函数
N_padding = [1024, 2048, 4096]; % 不同的零填充长度

for i = 1:length(N_padding)
    N = N_padding(i);
    X = fft(x_windowed, N);
    X_magnitude = abs(X/N);
    f = (0:N/2-1)*(fs/N);
    Pxx = (1/(fs*N)) * abs(X(1:N/2)).^2;
    subplot(3, 3, 3 + i);
    plot(f, 10*log10(Pxx), 'r');
    xlabel('频率 (Hz)', 'FontSize', 10);
    ylabel('功率谱密度 (dB/Hz)', 'FontSize', 10);
    title(['零填充长度 ' num2str(N)], 'FontSize', 12);
    grid on;
end

% -------- 重叠率的影响 --------
% 将信号分成多个段落，计算不同重叠率下的 PSD
overlap_ratios = [0.0, 0.5, 0.75];
window_length = 256;
n_overlap = [0, round(window_length*0.5), round(window_length*0.75)];

for i = 1:length(overlap_ratios)
    [S, f] = pwelch(x, hanning(window_length), n_overlap(i), N_padding(end), fs);
    subplot(3, 3, 6 + i);
    plot(f, 10*log10(S), 'g');
    xlabel('频率 (Hz)', 'FontSize', 10);
    ylabel('功率谱密度 (dB/Hz)', 'FontSize', 10);
    title(['重叠率 ' num2str(overlap_ratios(i)*100) '%'], 'FontSize', 12);
    grid on;
end


%窗口函数比较
%现在比较几种常用的窗口函数：Hanning 窗、Hamming 窗和 Blackman 窗。

clc
clear
close all

fs = 1000;           % 采样频率 (Hz)
t = 0:1/fs:1-1/fs;   % 时间序列
f1 = 50;             % 第一个频率分量 (Hz)
f2 = 150;            % 第二个频率分量 (Hz)
f3 = 300;            % 第三个频率分量 (Hz)

x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.2*sin(2*pi*f3*t);
x = x - mean(x);
windows = {'Hanning', 'Hamming', 'Blackman'};
window_functions = {hanning(length(x))', hamming(length(x))', blackman(length(x))'};
%%
figure;
for i = 1:length(windows)
    windowed_signal = x .* window_functions{i};
    N = 2^nextpow2(length(windowed_signal));
    X = fft(windowed_signal, N);
    X_magnitude = abs(X/N);
    f = (0:N/2-1)*(fs/N);
    Pxx = (1/(fs*N)) * abs(X(1:N/2)).^2;
    subplot(3, 4, 1 + (i-1)*4);
    plot(window_functions{i}, 'r');
    xlabel('样本点', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title([windows{i} ' 窗函数'], 'FontSize', 12);
    grid on;
    subplot(3, 4, 2 + (i-1)*4);
    plot(t, windowed_signal, 'g');
    xlabel('时间 (s)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title(['应用 ' windows{i} ' 窗后的信号'], 'FontSize', 12);
    grid on;
    subplot(3, 4, 3 + (i-1)*4);
    plot(f, 2*X_magnitude(1:N/2), 'k');
    xlabel('频率 (Hz)', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    title([windows{i} ' 窗 - FFT 频谱'], 'FontSize', 12);
    grid on;
    subplot(3, 4, 4 + (i-1)*4);
    plot(f, 10*log10(Pxx), 'm');
    xlabel('频率 (Hz)', 'FontSize', 10);
    ylabel('功率谱密度 (dB/Hz)', 'FontSize', 10);
    title([windows{i} ' 窗 - 功率谱密度'], 'FontSize', 12);
    grid on;
end
sgtitle('不同窗口函数对功率谱估计的影响分析', 'FontSize', 16);





