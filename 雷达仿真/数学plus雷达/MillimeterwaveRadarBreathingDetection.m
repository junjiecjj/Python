 
% https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247484149&idx=1&sn=8c3f0b1a97d26e434491734a7938e309&chksm=9780468547d18933a7b1e2b091353d7ef6f1bd4defbb6e01d705013c751bf086eecaa47481e3&mpshare=1&scene=1&srcid=1013vyPBXkDzec9rrJITvlO4&sharer_shareinfo=cb67f989932184dac10694a713cad499&sharer_shareinfo_first=cb67f989932184dac10694a713cad499&exportkey=n_ChQIAhIQKE4diKl4S%2BlQJ2IprJza6xKfAgIE97dBBAEAAAAAAE1vKadjvLgAAAAOpnltbLcz9gKNyK89dVj0u5xoH27A%2FvHbJCqAdCUJfEVbGET5gZhSk6roTaF6Q7Fvwfey8IBPhzimxIxEhzUILXbaFrz8Ekbl1aylKT%2FxClSC83ux09SDJ0rMVlbY%2FGzqRhHOwoyTQsURIxH4C%2BBEC9pcSu1FLle4zmDfSkMsaIawEgloDn%2BfSyr4DVjFM8y%2FFeUhzAofb6i2fq4GrjOsufYtlT%2BI9nDvdhn9AdJ%2B9kmBQrs7ZwNQo3aLnV8IrXhks7JSmcjaGb1%2B4EOKOEyS4g0kGZ1OIcICgQxRwcrBOv2E6d3KXrvtd4%2FqcBXcYlq2CXe4voRiUIAnfHLwvfMKDQmMSbCqbQLw&acctmode=0&pass_ticket=UXOzZPjQd8hKtQJfEhY5GKM4KzA6FuYbtPuIsR%2FM%2Bc1Zu7cs7uInoz8B68agQRtS&wx_header=0#rd
clear; close all; clc;

%% 优化参数设置
fs = 100;           % 采样频率 (Hz)
T = 90;             % 优化总时间
t = 0:1/fs:T-1/fs;  % 时间向量

% 雷达参数
fc = 60e9;          % 使用60GHz频段（更适合生命体征检测）
c = 3e8;
lambda = c/fc;

% 优化呼吸参数
f_breath = 0.25;    % 呼吸频率 15次/分钟
A_breath = 0.012;   % 增加呼吸幅度到12mm

% 优化心跳参数  
f_heart = 1.2;
A_heart = 0.0005;   % 增加心跳幅度

% 显著降低噪声
noise_level = 0.0003;

%% 1. 生成优化的生理信号
disp('生成优化的生理信号...');

% 基础呼吸信号 + 谐波
base_breath = A_breath * sin(2*pi*f_breath*t);
breath_harmonic = 0.08*A_breath * sin(4*pi*f_breath*t + pi/4);

% 心跳信号与呼吸调制
heart_signal_base = A_heart * sin(2*pi*f_heart*t);
heart_modulation = 0.15 * sin(2*pi*f_breath*t - pi/2); % 呼吸性心律不齐
heart_signal_modulated = heart_signal_base .* (1 + 0.1*heart_modulation);

% 组合信号
chest_displacement = base_breath + breath_harmonic + heart_signal_modulated;

% 最小化体动干扰
body_motion = zeros(size(t));
motion_times = [30, 60]; % 只在两个时间点有轻微体动
for mt = motion_times
    idx = find(t >= mt & t < mt + 2);
    if ~isempty(idx)
        motion_env = hanning(length(idx))';
        body_motion(idx) = 0.002 * sin(2*pi*0.8*(t(idx)-mt)) .* motion_env;
    end
end

% 最终位移信号
total_displacement = chest_displacement + body_motion + noise_level * randn(size(t));

%% 2. 优化的雷达信号生成
disp('生成优化的雷达信号...');

R0 = 1.0; % 缩短距离提高信噪比

% 相位信号生成
phase_signal = 4*pi*(R0 + total_displacement)/lambda;

% 极低相位噪声
phase_noise = 0.02 * randn(size(phase_signal));
phase_signal_clean = phase_signal + phase_noise;

radar_signal = exp(1j * phase_signal_clean);

%% 3. 高级信号处理
disp('进行高级信号处理...');

% 3.1 稳健的相位解缠
unwrapped_phase = unwrap(angle(radar_signal));

% 3.2 多项式去趋势
p = polyfit(t, unwrapped_phase, 2);
phase_trend = polyval(p, t);
detrended_phase = unwrapped_phase - phase_trend;

% 3.3 使用FIR滤波器（更稳定）
breath_filter = designfilt('bandpassfir', ...
                          'FilterOrder', 80, ...
                          'CutoffFrequency1', 0.18, ...
                          'CutoffFrequency2', 0.35, ...
                          'SampleRate', fs);

heart_filter = designfilt('bandpassfir', ...
                         'FilterOrder', 100, ...
                         'CutoffFrequency1', 0.9, ...
                         'CutoffFrequency2', 1.6, ...
                         'SampleRate', fs);

breath_signal = filtfilt(breath_filter, detrended_phase);
heart_signal = filtfilt(heart_filter, detrended_phase);

%% 4. 精确频率估计
disp('进行精确频率估计...');

N = length(t);
f = (0:N-1) * fs / N;

% 使用多窗口频谱估计
nw = 4; % 时间带宽积
[pxx, f_psd] = pmtm(detrended_phase, nw, N, fs);

% 呼吸频带寻找峰值
breath_mask = (f_psd >= 0.1) & (f_psd <= 0.5);
[~, breath_idx] = max(pxx(breath_mask));
f_breath_psd = f_psd(breath_mask);
detected_breath_freq = f_breath_psd(breath_idx);

% 心跳频带寻找峰值
heart_mask = (f_psd >= 0.8) & (f_psd <= 2.0);
[~, heart_idx] = max(pxx(heart_mask));
f_heart_psd = f_psd(heart_mask);
detected_heart_freq = f_heart_psd(heart_idx);

%% 5. 修正的多方法融合估计
disp('进行修正的多方法融合估计...');

% 方法1: 修正的自相关法
[acf, lags] = xcorr(breath_signal, 'coeff');
acf = acf(lags >= 0);
lags_sec = lags(lags >= 0) / fs; % 转换为秒

% 寻找主要峰值 - 修正：限制在合理呼吸周期范围内
min_period = 1/0.5; % 对应0.5Hz = 2秒周期
max_period = 1/0.1; % 对应0.1Hz = 10秒周期
valid_lags = (lags_sec >= min_period) & (lags_sec <= max_period);

if sum(valid_lags) > 0
    [peaks, peak_locs] = findpeaks(acf(valid_lags), 'MinPeakHeight', 0.3, 'MinPeakDistance', fs/4);
    if ~isempty(peaks)
        [~, main_peak_idx] = max(peaks);
        main_period = lags_sec(valid_lags);
        main_period = main_period(peak_locs(main_peak_idx));
        breath_freq_autocorr = 1 / main_period;
    else
        breath_freq_autocorr = detected_breath_freq;
    end
else
    breath_freq_autocorr = detected_breath_freq;
end

% 方法2: 改进的希尔伯特变换瞬时频率
analytic_signal = hilbert(breath_signal);
instantaneous_phase = unwrap(angle(analytic_signal));
instantaneous_freq = diff(instantaneous_phase) * fs / (2*pi);

% 过滤异常值
valid_freq_idx = (instantaneous_freq > 0.1) & (instantaneous_freq < 0.5);
if sum(valid_freq_idx) > 0
    breath_freq_hilbert = median(instantaneous_freq(valid_freq_idx));
else
    breath_freq_hilbert = detected_breath_freq;
end

% 方法3: 改进的零交叉法
zero_crossings = find(diff(sign(breath_signal - mean(breath_signal))) ~= 0);
if length(zero_crossings) >= 4
    % 计算完整周期（两个正到负或负到正的穿越）
    full_periods = [];
    for i = 2:length(zero_crossings)-1
        if sign(breath_signal(zero_crossings(i-1)+1)) > 0 && sign(breath_signal(zero_crossings(i+1)+1)) < 0
            period = (zero_crossings(i+1) - zero_crossings(i-1)) / fs;
            full_periods = [full_periods, period];
        end
    end
    if ~isempty(full_periods)
        breath_freq_zero = 1 / mean(full_periods);
    else
        breath_freq_zero = detected_breath_freq;
    end
else
    breath_freq_zero = detected_breath_freq;
end

% 智能加权融合（基于方法可靠性）
freq_estimates = [detected_breath_freq, breath_freq_autocorr, breath_freq_hilbert, breath_freq_zero];
reliability = ones(1,4);

% 评估每个方法的可靠性
% reliability(1) = pxx(breath_mask(breath_idx)) / max(pxx(breath_mask)); % MTM谱峰值高度
% reliability(2) = max(peaks)/max(acf(valid_lags)) if exist('peaks', 'var') && ~isempty(peaks) else 0.5; % 自相关峰值
% reliability(3) = 1 - mad(instantaneous_freq(valid_freq_idx))/median(instantaneous_freq(valid_freq_idx)) if sum(valid_freq_idx) > 10 else 0.5; % 希尔伯特稳定性
% reliability(4) = 1 - std(full_periods)/mean(full_periods) if exist('full_periods', 'var') && length(full_periods) >= 3 else 0.5; % 零交叉一致性

% 修正可靠性计算部分
reliability = ones(1,4);

% 评估每个方法的可靠性
reliability(1) = pxx(breath_mask(breath_idx)) / max(pxx(breath_mask)); % MTM谱峰值高度

% 自相关峰值可靠性
if exist('peaks', 'var') && ~isempty(peaks)
    reliability(2) = max(peaks) / max(acf(valid_lags));
else
    reliability(2) = 0.5;
end

% 希尔伯特稳定性可靠性
if sum(valid_freq_idx) > 10
    reliability(3) = 1 - mad(instantaneous_freq(valid_freq_idx)) / median(instantaneous_freq(valid_freq_idx));
else
    reliability(3) = 0.5;
end

% 零交叉一致性可靠性
if exist('full_periods', 'var') && length(full_periods) >= 3
    reliability(4) = 1 - std(full_periods) / mean(full_periods);
else
    reliability(4) = 0.5;
end

% 归一化可靠性权重
weights = reliability / sum(reliability);
breath_freq_final = sum(weights .* freq_estimates);

%% 6. 信号质量评估
disp('评估信号质量...');

% 计算真实的信噪比
breath_band_power = bandpower(breath_signal, fs, [0.15, 0.35]);
residual = detrended_phase - breath_signal - heart_signal;
noise_power = bandpower(residual, fs, [0.15, 0.35]);
if noise_power > 0
    snr_linear = breath_band_power / noise_power;
    snr_db = 10 * log10(snr_linear);
else
    snr_db = 30; % 如果噪声功率为0，设为高SNR
end

% 信号质量指标（0-100%）
if exist('peak_locs', 'var') && length(peak_locs) >= 3
    periods = diff(peak_locs);
    periodicity_score = 1 - std(periods) / mean(periods);
else
    periodicity_score = 0.6;
end

snr_score = min(1, snr_linear / 10); % 假设SNR>10为完美
consistency_score = 1 - std(freq_estimates) / mean(freq_estimates);

signal_quality = (0.4 * periodicity_score + 0.4 * snr_score + 0.2 * consistency_score) * 100;

%% 7. 最终可视化
disp('生成最终可视化...');

figure('Position', [50, 50, 1400, 900]);

% 7.1 信号分解图
subplot(2,3,1);
plot(t, total_displacement * 1000, 'Color', [0.2, 0.2, 0.8], 'LineWidth', 1);
hold on;
plot(t, base_breath * 1000, 'r', 'LineWidth', 2);
plot(t, heart_signal_modulated * 1000, 'g', 'LineWidth', 1.5);
plot(t, body_motion * 1000, 'm--', 'LineWidth', 1);
xlabel('时间 (秒)');
ylabel('位移 (mm)');
title('胸腔位移信号分解');
legend('总信号', '呼吸', '心跳', '体动', 'Location', 'best');
grid on;

% 7.2 多窗口频谱
subplot(2,3,2);
plot(f_psd, 10*log10(pxx), 'k', 'LineWidth', 2);
hold on;
xline(f_breath, 'r--', 'LineWidth', 2, 'Label', sprintf('真实呼吸 %.2fHz', f_breath));
xline(detected_breath_freq, 'r-', 'LineWidth', 2, ...
      'Label', sprintf('检测呼吸 %.2fHz', detected_breath_freq));
xline(f_heart, 'g--', 'LineWidth', 2, 'Label', sprintf('真实心跳 %.2fHz', f_heart));
xline(detected_heart_freq, 'g-', 'LineWidth', 2, ...
      'Label', sprintf('检测心跳 %.2fHz', detected_heart_freq));
xlabel('频率 (Hz)');
ylabel('功率谱密度 (dB/Hz)');
title('多窗口功率谱估计');
xlim([0, 2.5]);
grid on;

% 7.3 提取的呼吸信号
subplot(2,3,3);
plot(t, breath_signal, 'r', 'LineWidth', 2);
xlabel('时间 (秒)');
ylabel('幅度');
title('提取的呼吸信号');
grid on;

% 7.4 修正的自相关分析
subplot(2,3,4);
plot(lags_sec, acf, 'b', 'LineWidth', 2);
hold on;
if exist('peaks', 'var') && ~isempty(peaks)
    valid_lag_indices = find(valid_lags);
    plot(lags_sec(valid_lag_indices(peak_locs)), peaks, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
end
xlabel('延迟 (秒)');
ylabel('自相关系数');
title('呼吸信号自相关函数 (修正)');
xlim([0, 15]);
grid on;

% 7.5 多方法结果对比
subplot(2,3,5);
methods = {'MTM谱估计', '自相关法', '希尔伯特', '零交叉法', '最终融合'};
freq_values = [detected_breath_freq, breath_freq_autocorr, breath_freq_hilbert, breath_freq_zero, breath_freq_final];
errors = abs(freq_values - f_breath) * 60; % 转换为次/分钟误差

b = bar(freq_values * 60);
hold on;
yline(f_breath*60, 'r--', 'LineWidth', 3, 'Label', sprintf('真实值: %.1f次/分钟', f_breath*60));

% 在柱状图上标注误差和权重
for i = 1:length(freq_values)
    if i <= length(weights)
        weight_text = sprintf('权重: %.2f', weights(i));
    else
        weight_text = sprintf('权重: %.2f', weights(end));
    end
    text(i, freq_values(i)*60 + 0.5, sprintf('误差: %.2f\n%s', errors(i), weight_text), ...
         'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
end

set(gca, 'XTickLabel', methods);
ylabel('呼吸率 (次/分钟)');
title('多方法检测结果对比 (修正)');
ylim([0, 20]);
grid on;

% 7.6 性能仪表盘
subplot(2,3,6);

% 创建性能指标显示
performance_metrics = {
    '频率精度', '信噪比', '周期性', ...
    '一致性', '算法可靠性'
};

metric_values = [
    max(0, 100 - errors(end)*10),   % 频率精度 (误差越小越好)
    min(100, max(0, snr_db + 20)),  % 信噪比
    periodicity_score * 100,         % 周期性
    consistency_score * 100,         % 一致性
    mean(reliability) * 100          % 平均可靠性
];

% 简单条形图显示性能
barh(metric_values);
set(gca, 'YTickLabel', performance_metrics);
xlabel('性能分数 (%)');
xlim([0, 100]);
title('系统性能评估');
grid on;

for i = 1:length(metric_values)
    text(metric_values(i)+2, i, sprintf('%.1f%%', metric_values(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 10);
end

sgtitle('毫米波雷达呼吸检测 - 最终修正版本', 'FontSize', 16, 'FontWeight', 'bold');

%% 8. 详细结果输出
fprintf('\n=== 最终修正版本结果 ===\n');
fprintf('真实呼吸率: %.3f Hz (%.1f 次/分钟)\n', f_breath, f_breath*60);
fprintf('MTM谱估计:  %.3f Hz (%.1f 次/分钟) - 误差: %.2f 次/分钟\n', ...
        detected_breath_freq, detected_breath_freq*60, errors(1));
fprintf('自相关法:   %.3f Hz (%.1f 次/分钟) - 误差: %.2f 次/分钟\n', ...
        breath_freq_autocorr, breath_freq_autocorr*60, errors(2));
fprintf('希尔伯特:   %.3f Hz (%.1f 次/分钟) - 误差: %.2f 次/分钟\n', ...
        breath_freq_hilbert, breath_freq_hilbert*60, errors(3));
fprintf('零交叉法:   %.3f Hz (%.1f 次/分钟) - 误差: %.2f 次/分钟\n', ...
        breath_freq_zero, breath_freq_zero*60, errors(4));
fprintf('最终结果:   %.3f Hz (%.1f 次/分钟) - 误差: %.2f 次/分钟\n', ...
        breath_freq_final, breath_freq_final*60, errors(5));
fprintf('系统信噪比: %.2f dB\n', snr_db);
fprintf('信号质量:   %.1f%%\n', signal_quality);
fprintf('方法可靠性: [%.2f, %.2f, %.2f, %.2f]\n', reliability);

if errors(5) <= 1.0
    fprintf('检测结果: ✓ 优秀 (临床级精度)\n');
elseif errors(5) <= 2.0
    fprintf('检测结果: ✓ 良好 (实用级精度)\n');
elseif errors(5) <= 3.0
    fprintf('检测结果: ○ 一般 (需进一步优化)\n');
else
    fprintf('检测结果: △ 需要改进\n');
end