

clear; close all; clc;
%% 1.0 参数初始化
c = 3e8;                               % 光速
fc = 77e9;                             % 中心频率
lambda = c / fc;                       % 参考波长（77GHz）
N_tx = 4;                              % 发射天线数
N_rx = 4;                              % 接收天线数
N_virtual_array = N_tx * N_rx;         % 虚拟阵列天线数
Tc = 30e-6;                            % 脉冲重复时间
Fs = 40e6;                             % 采样率
Ts = 1/Fs;                             % 采样周期
Bandwidth = 500e6;                     % 射频带宽
Slope = 30e12;                         % 调频斜率(MHz/us)
Samples_per_chirp = 512;               % 每个chirp采样点数
Num_chirp_total = 1;                   % 总chirp数量
Tchirp_Valid = Samples_per_chirp * Ts;          % 采样总时间
BandwidthValid = Slope * Tchirp_Valid; % 有效带宽
t = 0 : Ts: (Samples_per_chirp-1) * Ts;             % 快时间轴, 实际ADC采样的时刻, t = (0 : N_sample-1) * Ts

% 1.2 角反射器空间位置参数
R_true = 10.0;                                   % 真实距离（m）
v_true = 0.0;                                    % 真实速度（m/s）
angles_cal = -75 : 5 : 75;                       % 方位角（度）
SNR = 15;                                        % SNR(dB) 
Num_angle = length(angles_cal);                  % 目标数量

% 1.3 波形参数
TidleTime_Fastchirp = 5e-6;                             % 波形的静默时间
Tchirp_Valid_Fastchirp = Samples_per_chirp * Ts;        % 数字处理有效发波时间
Tc_Fastchirp = Tc + TidleTime_Fastchirp;                % 波形的重复周期

% 1.4 阵列天线参数（固定物理位置-米制单位）
d_TX = [0, 2, 4, 6] * lambda;             % 发射天线位置（米）
d_RX = [0, 0.5, 1.0, 1.5] * lambda;       % 接收天线位置（米）
d = lambda/2;                             % 虚拟阵元间距 (m)

% 1.5 数字信号处理
range_fft_bin = 512;        % 距离维度fft点数

% 1.6 计算距离分辨率
range_res = c / (2 * BandwidthValid);

%% 2.0 生成预设通道误差（幅度和相位）
% 2.1 幅度误差范围：-3 dB 到 +6 dB(线性单位：[0.7, 1.4])
gain_errors = 0.7 + 1.4 * rand(1, N_virtual_array); 

% 2.2 相位误差范围：-30° 到 30°
phase_errors = -30 + 60 * rand(1, N_virtual_array); 
 
%% 14.0 目标方位角测量功能
% 14.1 添加两个目标：目标1和目标2
target_angles = [-50, 10]; % 目标方位角（度）
target_R = 10; % 目标距离（m）
target_v = 0; % 目标速度（m/s）

% 14.2 生成目标信号
target_signals = zeros(length(target_angles), Samples_per_chirp, N_virtual_array);
t_abs = 1
for angle_idx = 1 : length(target_angles)
    fd = 2 * target_v / lambda;       % 多普勒频率
    tau = 2 * target_R / c;           % 目标延迟时间

    % 生成发射信号
    tx_signal = exp(1j * 2 * pi * (fc * (t + t_abs) + 0.5 * Slope * t.^2));

    for tx_idx = 1 : N_tx
        for rx_idx = 1 : N_rx
            % 计算空间参数
            virt_pos = d_TX(tx_idx) + d_RX(rx_idx);

            % 中频相位模型
            phase_if =  2 * pi * ((fc - fd) * (t - tau + t_abs)) + pi * Slope * (t - tau).^2;

            % 动态阵列相位模型
            array_phase = (2 * pi / lambda) * virt_pos * sind(target_angles(angle_idx));

            % 通道误差信号
            virt_idx = (tx_idx - 1) * N_rx + rx_idx;
            gain_phase_errors = gain_errors(virt_idx) * exp(1j * deg2rad(phase_errors(virt_idx)));

            % 接收信号 + 通道误差信号合成
            rx_signal = exp(1j * (phase_if - array_phase)) * gain_phase_errors;

            % 混频
            if_signal = tx_signal.* conj(rx_signal);

            target_signals(angle_idx, :, virt_idx) = if_signal;
        end
    end
end

% 14.3 添加噪声
target_signals = awgn(target_signals, SNR, 'measured');

% 14.4 距离维FFT处理
target_signals_windowed = target_signals .* win_range;
target_range_fft = fft(target_signals_windowed, range_fft_bin, 2);

% 14.5 提取目标处的距离维FFT复数
X_k0_target = zeros(N_virtual_array, length(target_angles));

for a_idx = 1 : length(target_angles)
    for ch = 1 : N_virtual_array
        [~, range_fft_peakidx] = max(abs(target_range_fft(a_idx, :, ch)));
        X_k0_target(ch, a_idx) = target_range_fft(a_idx, range_fft_peakidx, ch);
    end
end

% 角度FFT参数
angle_fft_bin = 512;
angle_range = -90:180/(angle_fft_bin-1):90; % 角度范围

% 14.6 校准前的角度FFT
angle_fft_before = zeros(angle_fft_bin, length(target_angles));
for t_idx = 1:length(target_angles)
    % 构建阵列流形向量
    array_response = zeros(angle_fft_bin, N_virtual_array);
    for ang_idx = 1:angle_fft_bin
        for ch = 1:N_virtual_array
            spatial_phase = (2 * pi / lambda) * (ch - 1) * d * sind(angle_range(ang_idx));
            array_response(ang_idx, ch) = exp(-1j * spatial_phase);
        end
    end

    % 14.7 计算角度FFT
    for ang_idx = 1:angle_fft_bin
        angle_fft_before(ang_idx, t_idx) = abs(array_response(ang_idx, :) * X_k0_target(:, t_idx));
    end
end

% 校准后的角度FFT
X_k0_target_comp = zeros(size(X_k0_target));
for t_idx = 1:length(target_angles)
    for ch = 1:N_virtual_array
        % 应用校准系数
        X_k0_target_comp(ch, t_idx) = X_k0_target(ch, t_idx) * comp_gain(ch) * exp(1j * deg2rad(comp_phase(ch)));
    end
end

angle_fft_after = zeros(angle_fft_bin, length(target_angles));
for t_idx = 1:length(target_angles)
    % 构建阵列流形向量
    array_response = zeros(angle_fft_bin, N_virtual_array);
    for ang_idx = 1:angle_fft_bin
        for ch = 1:N_virtual_array
            spatial_phase = (2 * pi / lambda) * (ch - 1) * d * sind(angle_range(ang_idx));
            array_response(ang_idx, ch) = exp(-1j * spatial_phase);
        end
    end

    % 计算角度FFT
    for ang_idx = 1:angle_fft_bin
        angle_fft_after(ang_idx, t_idx) = abs(array_response(ang_idx, :) * X_k0_target_comp(:, t_idx));
    end
end

% 归一化角度谱
angle_fft_before = angle_fft_before ./ max(angle_fft_before(:));
angle_fft_after = angle_fft_after ./ max(angle_fft_after(:));
 
%% 可视化结果 - 目标角度测量比对
figure('Name', '目标角度测量校准前后比对', 'Position', [200, 100, 1000, 600]);

% 校准前的角度谱
subplot(2, 2, 1);
plot(angle_range, 20*log10(angle_fft_before(:, 1)), 'b', 'LineWidth', 2);
hold on;
plot(angle_range, 20*log10(angle_fft_before(:, 2)), 'r', 'LineWidth', 2);
xline(target_angles(1), '--b', '目标1', 'LabelVerticalAlignment', 'middle');
xline(target_angles(2), '--r', '目标2', 'LabelVerticalAlignment', 'middle');
title('校准前角度谱');
xlabel('方位角 (°)');
ylabel('归一化幅度 (dB)');
xlim([-90, 90]);
ylim([-40, 0]);
grid on;
legend('目标1', '目标2');
set(gca, 'FontSize', 12);

% 校准后的角度谱
subplot(2, 2, 2);
plot(angle_range, 20*log10(angle_fft_after(:, 1)), 'b', 'LineWidth', 2);
hold on;
plot(angle_range, 20*log10(angle_fft_after(:, 2)), 'r', 'LineWidth', 2);
xline(target_angles(1), '--b', '目标1', 'LabelVerticalAlignment', 'middle');
xline(target_angles(2), '--r', '目标2', 'LabelVerticalAlignment', 'middle');
title('校准后角度谱');
xlabel('方位角 (°)');
ylabel('归一化幅度 (dB)');
xlim([-90, 90]);
ylim([-40, 0]);
grid on;
legend('目标1', '目标2');
set(gca, 'FontSize', 12);

% 校准前峰值角度提取
[~, peak_idx1_before] = max(angle_fft_before(:, 1));
[~, peak_idx2_before] = max(angle_fft_before(:, 2));
peak_angle1_before = angle_range(peak_idx1_before);
peak_angle2_before = angle_range(peak_idx2_before);

% 校准后峰值角度提取
[~, peak_idx1_after] = max(angle_fft_after(:, 1));
[~, peak_idx2_after] = max(angle_fft_after(:, 2));
peak_angle1_after = angle_range(peak_idx1_after);
peak_angle2_after = angle_range(peak_idx2_after);

% 角度测量误差比对
subplot(2, 2, 3);
error_before = [abs(peak_angle1_before - target_angles(1)), abs(peak_angle2_before - target_angles(2))];
error_after =  [abs(peak_angle1_after  - target_angles(1)), abs(peak_angle2_after -  target_angles(2))];

bar([error_before; error_after]');
set(gca, 'XTickLabel', {'目标1', '目标2'});
ylabel('角度测量误差 (°)');
title('角度测量误差比对');
legend('校准前', '校准后');
grid on;
set(gca, 'FontSize', 12);

% 旁瓣电平比对
subplot(2, 2, 4);

% 动态计算主瓣宽度（基于阵列理论）
N = N_virtual_array; % 虚拟阵元数
d = lambda/2; % 阵元间距
% 对于每个目标角度
for t_idx = 1:length(target_angles)
    theta = target_angles(t_idx);
    hpbw = 0.886 * lambda / (N * d * cosd(theta)) * (180/pi); % HPBW in degrees
    mainlobe_width = 2 * hpbw; % 使用2倍HPBW作为主瓣宽度

    % 创建掩码，排除主瓣区域
    sidelobe_mask = ~(angle_range > (theta - mainlobe_width/2) & angle_range < (theta + mainlobe_width/2));

    % 计算最高旁瓣电平
    sidelobe_level_before(t_idx) = max(20*log10(angle_fft_before(sidelobe_mask, t_idx)));
    sidelobe_level_after(t_idx) = max(20*log10(angle_fft_after(sidelobe_mask, t_idx)));
end

bar([sidelobe_level_before; sidelobe_level_after]');
set(gca, 'XTickLabel', {'目标1', '目标2'});
ylabel('最高旁瓣电平 (dB)');
title('旁瓣电平比对');
legend('校准前', '校准后');
grid on;
set(gca, 'FontSize', 12);

% 输出测量结果
fprintf('\n6.0 目标角度测量结果:\n');
fprintf('校准前 - 目标1: 测量角度=%.2f°, 误差=%.2f°, 旁瓣电平=%.2f dB\n', ...
        peak_angle1_before, error_before(1), sidelobe_level_before(1));
fprintf('校准前 - 目标2: 测量角度=%.2f°, 误差=%.2f°, 旁瓣电平=%.2f dB\n', ...
        peak_angle2_before, error_before(2), sidelobe_level_before(2));
fprintf('校准后 - 目标1: 测量角度=%.2f°, 误差=%.2f°, 旁瓣电平=%.2f dB\n', ...
        peak_angle1_after, error_after(1), sidelobe_level_after(1));
fprintf('校准后 - 目标2: 测量角度=%.2f°, 误差=%.2f°, 旁瓣电平=%.2f dB\n', ...
        peak_angle2_after, error_after(2), sidelobe_level_after(2));