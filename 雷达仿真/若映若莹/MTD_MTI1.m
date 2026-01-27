% 仿真二：FFT方法设计一个16通道的多普勒滤波器组，来进行MTD。




%% 基于FFT多普勒滤波器组的16通道MTD处理仿真
clear; clc; close all;

%% ===================== 6. 辅助函数：快速卷积（FFT实现） =====================
function y = fftconv(x, h, mode)
    % FFT实现快速卷积，避免时域卷积的高计算量
    N = length(x) + length(h) - 1;
    X = fft(x, N);
    H = fft(h, N);
    y = ifft(X .* H);
    if nargin == 3 && strcmp(mode, 'full')
        % 输出完整卷积结果
    elseif strcmp(mode, 'same')
        y = y(ceil(N/2)-floor(length(x)/2):ceil(N/2)+floor(length(x)/2));
    end
end

%% ===================== 1. 核心参数设置 =====================
fc = 10e9;             % 载波频率10GHz
c = 3e8;               % 光速(m/s)
lambda = c/fc;         % 波长(m)
fd_target = 2.5e3;     % 目标多普勒频率2.5kHz
PRF = 10e3;            % 脉冲重复频率10kHz（满足PRF>2*fd_target，无混叠）
N_filter = 16;         % 多普勒滤波器组数量（16通道）
N_pulse = 128;         % 慢时间脉冲数（滤波器长度）
SNR = 10;              % 回波信噪比(dB)
fs = PRF;              % 慢时间采样率(Hz)
%% ===================== 2. 生成16通道回波数据 =====================
% 慢时间轴（脉冲维度）
t_slow = (0:N_pulse-1)/fs;  
% 初始化回波矩阵：行=脉冲数，列=原始通道数（先生成1路目标回波，再扩展为16通道）
echo_raw = zeros(N_pulse, N_filter);  
% 生成含多普勒的基带回波（叠加高斯噪声）
signal_clean = exp(1j*2*pi*fd_target*t_slow);  % 复正弦多普勒信号
for ch = 1:N_filter
    echo_raw(:, ch) = awgn(signal_clean, SNR, 'measured');  % 16通道回波（同目标，不同噪声）
end
%% ===================== 3. 设计FFT多普勒滤波器组 =====================
% 步骤1：生成频域滤波器系数（16个滤波器对应16个多普勒频率点）
fd_bins = (-N_filter/2:N_filter/2-1)*(fs/N_filter);  % 滤波器组对应的多普勒频率点(Hz)
fft_filters = zeros(N_pulse, N_filter);  % 滤波器系数矩阵（行=脉冲数，列=滤波器通道）
% 步骤2：基于FFT构建滤波器（加汉宁窗降低旁瓣）
window = hanning(N_pulse)';  % 窗函数
for n = 1:N_filter
    % 第n个滤波器的频域中心频率
    fd_center = fd_bins(n);  
    % 时域滤波器系数：复指数调谐 + 窗函数
    fft_filters(:, n) = window .* exp(1j*2*pi*fd_center*t_slow);  
end
% 步骤3：对滤波器做共轭，用于相关滤波（匹配滤波）
fft_filters = conj(fft_filters);  
%% ===================== 4. 执行MTD滤波（卷积/相关运算） =====================
mtd_output = zeros(N_pulse, N_filter);  % MTD滤波输出（幅度谱）
fd_axis = (-N_pulse/2:N_pulse/2-1)*(fs/N_pulse);  % 最终多普勒频率轴
% 对每个通道回波执行多普勒滤波
for ch = 1:N_filter
    % 方法：用第ch个滤波器对第ch个通道回波做线性卷积（保留完整长度）
    filter_out = fftconv(echo_raw(:, ch), fft_filters(:, ch), 'full');  
    % 取稳态段（长度=N_pulse）并做FFT移位，得到多普勒谱
    filter_out = filter_out(N_pulse/2:end-N_pulse/2);  % 截取有效段
    fft_out = fftshift(fft(filter_out));  % FFT并移位（零频居中）
    % 归一化为dB幅度
    mtd_output(:, ch) = 20*log10(abs(fft_out)/max(abs(fft_out)));  
end
%% ===================== 5. 绘制16通道MTD结果图 =====================
figure
for ch = 1:N_filter/2
    subplot(2,4,ch)
    % 绘制当前通道的多普勒幅度谱
    plot(fd_axis, mtd_output(:, ch), 'b-', 'LineWidth', 1.2);
    hold on;
    % 标记目标多普勒频率（红色虚线）
    plot([fd_target, fd_target], ylim, 'r--', 'LineWidth', 1.5);
    % 标记当前滤波器中心频率（黑色点线）
    plot([fd_bins(ch), fd_bins(ch)], ylim, 'k:', 'LineWidth', 1);

    xlabel('多普勒频率 (Hz)');
    ylabel('幅度 (dB)');
    title(sprintf('通道%d (中心频率%.1fHz)', ch, fd_bins(ch)));
    grid on;
    xlim([-fs/2, fs/2]);  % Nyquist频率范围
    ylim([-60, 0]);       % 幅度范围（突出主瓣）
    legend('MTD谱', '目标频率', '滤波器中心', 'Location', 'best');
end
figure
for ch = N_filter/2+1:N_filter
    n = ch - N_filter/2;
    subplot(2,4,n);
    % 绘制当前通道的多普勒幅度谱
    plot(fd_axis, mtd_output(:, ch), 'b-', 'LineWidth', 1.2);
    hold on;
    % 标记目标多普勒频率（红色虚线）
    plot([fd_target, fd_target], ylim, 'r--', 'LineWidth', 1.5);
    % 标记当前滤波器中心频率（黑色点线）
    plot([fd_bins(ch), fd_bins(ch)], ylim, 'k:', 'LineWidth', 1);

    xlabel('多普勒频率 (Hz)');
    ylabel('幅度 (dB)');
    title(sprintf('通道%d (中心频率%.1fHz)', ch, fd_bins(ch)));
    grid on;
    xlim([-fs/2, fs/2]);  % Nyquist频率范围
    ylim([-60, 0]);       % 幅度范围（突出主瓣）
    legend('MTD谱', '目标频率', '滤波器中心', 'Location', 'best');
end





