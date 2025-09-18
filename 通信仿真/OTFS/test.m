
clc; clear; close all;
%% 1，生成多个周期的脉冲信号，脉内调制为线性调频信号。
% 1. 参数设置
T = 20e-6;            % 单个脉冲宽度 (s)
B = 20e6;             % 调频带宽 (Hz)
fs = 10 * B;           % 采样率，奈奎斯特采样，取为 2B
N_pulse = 5;         % 脉冲数
DutyCycle = 0.1;      % 占空比 (10%)
PRI = T / DutyCycle;  % 脉冲重复间隔 (s)
N_total = round(PRI * fs * N_pulse); % 总采样点数
t = (0:N_total-1) / fs; % 总时间序列

% 2. 生成单个 LFM 脉冲信号
alpha = B / T;        % 调频斜率
samples_pulse = round(T * fs); % 单个脉冲的采样点数
t_pulse = (0:samples_pulse-1) / fs; % 单个脉冲的时间序列
s_pulse = exp(1j * pi * alpha * t_pulse.^2); % 单个脉冲信号

% 3. 生成脉冲列
s_tx = zeros(1, N_total);
samples_gap = round(PRI * fs) - samples_pulse; % 脉冲间隔的采样点数

if samples_gap < 0
    error('占空比过高，导致脉冲间隔为负。请调整占空比或脉冲宽度。');
end

for n = 0:N_pulse-1
    start_idx = n * (samples_pulse + samples_gap) + 1;
    end_idx = start_idx + samples_pulse -1;
    if end_idx <= N_total
        s_tx(start_idx:end_idx) = s_pulse;
else
        warning('脉冲 %d 超出总采样点数范围，未完全生成。', n+1);
    end
end

% 可视化发射脉冲列
figure;
plot(t*1e6, real(s_tx));
title('发射 LFM 脉冲列（实部）'); xlabel('时间 (µs)'); ylabel('幅度');
axis([0, PRI*1e6*N_pulse, -1.2, 1.2]);


%% 2，根据多个目标的延时生成目标的叠加回波，并添加噪声。

% 4. 模拟多个目标的回波
% 定义多个目标，每个目标有时延 Tau 和衰减系数 A
targets = [
    struct('Tau', 5e-6, 'A', 1.0),
    struct('Tau', 15e-6, 'A', 0.8),
    struct('Tau', 25e-6, 'A', 0.6)
];

rcv_sig = zeros(1, N_total);

for k = 1:length(targets)
    Tau = targets(k).Tau;
    A = targets(k).A;
    delay_samples = round(Tau * fs)

if delay_samples >= N_total
        warning('目标 %d 的时延超过总信号长度，忽略该目标。', k);
        continue;
end

    % 将发射信号延时后叠加到接收信号中
    rcv_sig(delay_samples+1:end) = rcv_sig(delay_samples+1:end) + A * s_tx(1:N_total-delay_samples);
end

% 添加噪声（可选）
SNR_dB = 20; % 信噪比
signal_power = rms(rcv_sig).^2;
noise_power = signal_power / (10^(SNR_dB/10));
noise = sqrt(noise_power/2) * (randn(1, N_total) + 1j*randn(1, N_total));
rcv_sig_noisy = rcv_sig + noise;

figure;
plot(t*1e6, abs(rcv_sig_noisy),'r');
title('接收回波幅度（含噪声）'); xlabel('时间 (µs)'); ylabel('幅度');
hold on;
plot(t*1e6, real(rcv_sig_noisy),'b');
hold off;

%% 3，进行匹配滤波（快速卷积实现）

% 5. 匹配滤波器
h = conj(fliplr(s_pulse));

% 6. 快速卷积实现匹配滤波
len_fft = 2^nextpow2(length(rcv_sig_noisy) + length(h) - 1);
S_fft = fft(rcv_sig_noisy, len_fft);
H_fft = fft(h, len_fft);
y_fft = S_fft .* H_fft;
mf_output = ifft(y_fft);

% 截取有效长度
mf_output = mf_output(1:length(rcv_sig_noisy));

% 7. 可视化匹配滤波输出
figure;
plot(t*1e6, 20*log10(abs(mf_output)));
title('匹配滤波输出幅度'); xlabel('时间 (µs)'); ylabel('幅度');


