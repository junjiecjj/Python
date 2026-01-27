% 仿真三：FIR方法设计一个16通道的多普勒滤波器组，来进行MTD

%% 基于FIR滤波器组的16通道MTD处理仿真（解决所有报错）
clear; clc; close all;
%% ===================== 1. 核心参数配置（关键：匹配数据长度和滤波器阶数） =====================
% 雷达基础参数
fc = 10e9;             % 载波频率10GHz
c = 3e8;               % 光速(m/s)
lambda = c/fc;         % 波长(m)
fd_target = 2.5e3;     % 目标多普勒频率2.5kHz
PRF = 10e3;            % 脉冲重复频率10kHz（无多普勒混叠）
fs = PRF;              % 慢时间采样率(Hz) = PRF
nyq_freq = fs/2;       % 奈奎斯特频率(5kHz)
% 滤波器组参数（核心修正：FIR阶数≤数据长度/3）
fir_order = 64;        % FIR滤波器阶数（64，3倍为192 < 数据长度400）
N_channel = 16;        % 多普勒滤波器组通道数（16个）
SNR = 10;              % 回波信噪比(dB)
N_pulse = 400;         % 慢时间脉冲数（数据长度≥3*fir_order=192，设为400留余量）
% 多普勒频率轴（Nyquist区间：-nyq_freq ~ nyq_freq）
fd_range = [-nyq_freq, nyq_freq];
% 16个滤波器的中心频率（均匀分布，避开边界）
fd_centers = linspace(fd_range(1) + 10, fd_range(2) - 10, N_channel);
% 每个滤波器的通带宽度（均匀划分）
fd_bandwidth = (fd_range(2) - fd_range(1) - 20)/N_channel;
%% ===================== 2. 生成16通道回波数据（长度匹配） =====================
% 慢时间轴
t_slow = (0:N_pulse-1)/fs;  
% 生成含目标多普勒的基带回波（复信号）
signal_clean = exp(1j*2*pi*fd_target*t_slow);  % 纯多普勒信号
echo_data = zeros(N_pulse, N_channel);         % 16通道回波矩阵
% 为每个通道叠加不同高斯噪声
for ch = 1:N_channel
    echo_data(:, ch) = awgn(signal_clean, SNR, 'measured');
end
%% ===================== 3. 设计16通道FIR多普勒滤波器组（修正版） =====================
fir_filters = cell(1, N_channel);  % 存储每个通道的FIR滤波器系数
dev = [0.01 0.001];  % 通带/阻带波纹
% 逐个设计16个FIR带通滤波器
for ch = 1:N_channel
    % 当前滤波器的通带范围（严格限制在Nyquist区间内）
    fd_pass_low = fd_centers(ch) - fd_bandwidth/2;
    fd_pass_high = fd_centers(ch) + fd_bandwidth/2;

    % 强制限制通带边缘不越界
    fd_pass_low = max(fd_pass_low, -nyq_freq + 1);
    fd_pass_high = min(fd_pass_high, nyq_freq - 1);

    % 归一化频率（0~1，对应0~fs/2）
    f_pass = [abs(fd_pass_low), abs(fd_pass_high)] / nyq_freq;
    if fd_centers(ch) < 0
        f_pass = fliplr(f_pass);  % 负频率带通交换高低端
    end

    % 设计FIR滤波器（优先firpm，失败则用fir1）
    try
        [N, Fo, Ao, W] = firpmord(f_pass, [1 1], dev, 1);
        b = firpm(min(N, fir_order), Fo, Ao, W);
    catch
        b = fir1(fir_order, f_pass, 'bandpass', hanning(fir_order+1));
    end

    % 负频率滤波器复调制
    if fd_centers(ch) < 0
        t_filter = (0:length(b)-1)/fs;
        b = b .* exp(-1j*2*pi*nyq_freq*t_filter);
    end

    fir_filters{ch} = b;
end
%% ===================== 4. 执行MTD滤波（filtfilt无报错） =====================
mtd_result = zeros(N_pulse, N_channel);  % MTD输出结果
fd_axis = (-N_pulse/2:N_pulse/2-1)*(fs/N_pulse);  % 多普勒频率轴
for ch = 1:N_channel
    % 零相位滤波（数据长度400 > 3*64=192，无报错）
    echo_filtered = filtfilt(fir_filters{ch}, 1, echo_data(:, ch));

    % FFT并计算归一化dB幅度（加eps避免除零）
    fft_data = fftshift(fft(echo_filtered));
    amp_dB = 20*log10(abs(fft_data)/(max(abs(fft_data)) + eps));
    mtd_result(:, ch) = amp_dB;
end
%% ===================== 5. 绘制16通道MTD结果图 =====================
figure('Position', [50, 50, 1200, 800]);
for ch = 1:N_channel
    subplot(4, 4, ch);
    % 绘制MTD幅度谱
    plot(fd_axis, mtd_result(:, ch), 'b-', 'LineWidth', 1.2);
    hold on;
    % 标记目标多普勒频率
    plot([fd_target, fd_target], ylim, 'r--', 'LineWidth', 1.5);
    % 标记滤波器中心频率
    plot([fd_centers(ch), fd_centers(ch)], ylim, 'k:', 'LineWidth', 1);

    % 图形美化
    xlabel('多普勒频率 (Hz)');
    ylabel('幅度 (dB)');
    title(sprintf('通道%d (中心频率%.1fHz)', ch, fd_centers(ch)));
    grid on;
    xlim(fd_range);
    ylim([-60, 0]);
    legend('MTD谱', '目标频率', '滤波器中心', 'Location', 'best');
end
sgtitle('16通道FIR多普勒滤波器组MTD结果（目标多普勒2.5kHz）', 'FontSize', 14);
%% ===================== 6. 滤波器特性验证 =====================
% 找到目标频率对应的通道
ch_selected = find(abs(fd_centers - fd_target) == min(abs(fd_centers - fd_target)));
[h, f] = freqz(fir_filters{ch_selected}, 1, 1024, fs);
figure('Position', [100, 100, 800, 400]);
subplot(1,2,1);
plot(f, 20*log10(abs(h)));
xlabel('频率 (Hz)');
ylabel('幅度 (dB)');
title(sprintf('通道%d FIR滤波器幅频响应', ch_selected));
grid on;
xlim(fd_range);
ylim([-80, 10]);
% 绘制滤波前后时域波形
subplot(1,2,2);
plot(t_slow, real(echo_data(:, ch_selected)), 'b-', 'LineWidth', 1);
hold on;
echo_filtered = filtfilt(fir_filters{ch_selected}, 1, echo_data(:, ch_selected));
plot(t_slow, real(echo_filtered), 'r-', 'LineWidth', 1);
xlabel('慢时间 (s)');
ylabel('幅度');
title('滤波前后时域波形');
legend('滤波前', '滤波后');
grid on;