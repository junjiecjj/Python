


%% 系统参数设置
% 1.0 系统参数
freq = 77e9;                  % 工作频率 [Hz]
c = physconst('LightSpeed');  % 光速 [m/s]
lambda = c / freq;            % 波长 [m]
R = 320;                      % 目标距离 [m]
sigma = 10;                   % 目标RCS [m²] (保守值)
N_tx = 4;                     % 发射天线数量
N_rx = 8;                     % 接收天线数量

% 2.0 系统损耗 (dB值)
L_sys_dB = 6;                 % 总系统损耗 (馈线/开关/处理等)
L_sys = 10^(L_sys_dB/10);     % 转换为线性值

% 3.0 天线设计参数
G_ant_dB = 13.5;                % 单天线增益 [dBi]
G_ant = 10^(G_ant_dB/10);       % 转换为线性值
%G_array = N_tx * N_rx;          % 阵列增益因子 (线性值)
G_array = N_rx;
G_array_dB = 10*log10(G_array); % 阵列增益 [dB]

% 4.0 波形设计参数
B_mod = 240e6;                  % 波形调制带宽 [Hz] (决定距离分辨率)
T_chirp = 20e-6;                % 调频脉宽 [s]
N_p = 250;                      % 积累脉冲数

% 5.0 CFAR检测性能要求
SNR_min_dB = 13;                 % 所需最小输出SNR [dB]
SNR_min = 10^(SNR_min_dB/10);    % 转换为线性值

% 6.0 接收机设计参数
B_rec = 1.1*B_mod;               % 接收机噪声带宽 [Hz] (通常设置等于调制带宽), 实际系统可能略大(1.1-1.2倍), 影响噪声功率的估计
NF_dB = 12;                       % 噪声系数 [dB]
NF = 10^(NF_dB/10);                % 噪声系数 (线性)
k = 1.38e-23;                    % 玻尔兹曼常数
T0 = 290;                        % 标准噪声温度 [K]

% 7.0 接收机噪声参数
Pn = k * T0 * B_rec * NF;         % 噪声功率 [W] (关键修正：使用B_rec)
Pn_dBm = 10*log10(Pn/1e-3);      % 噪声功率 [dBm]

% 8.0 信号处理增益计算
G_pulse_comp = B_mod * T_chirp;    % 脉冲压缩增益 (线性)
G_coh_int = N_p;                   % 相干积累增益 (线性)
G_proc = G_pulse_comp * G_coh_int; % 总处理增益 (线性)
G_proc_dB = 10*log10(G_proc);      % 总处理增益 [dB]

% 1.1 雷达参数
c = 3e8;                               % 光速
fc = 77e9;                             % 中心频率
T_RF = 20e-6;                          % 射频发波时间
Fs = 40e6;                             % 采样率
Ts = 1/Fs;                             % 采样周期
Slope = 18.5e12;                         % 调频斜率(MHz/us)
Bandwidth = Slope * T_RF;              % 射频发射带宽
Samples_per_chirp = 512;               % 每个chirp采样点数
Num_chirp_total = 500;                 % 总chirp数量
Num_Fastchirp = Num_chirp_total / 2;
Num_Slowchirp = Num_chirp_total / 2;
Tchirp_Valid = Samples_per_chirp * Ts; % 采样总时间
BandwidthValid = Slope * Tchirp_Valid; % 有效带宽
t = 0 : Ts: (Samples_per_chirp-1) * Ts;% 快时间轴, 实际ADC采样的时刻, t = (0 : N_sample-1) * Ts
centerFreq = fc + Bandwidth/2;         % 载波中心频率
lambda = c / centerFreq;               % 参考波长（77GHz）

% 1.2 目标参数
R_true = [10.8 110.5 188.9]';     % 真实距离（m）
v_true = [112.5 102.5 -55.5]';   % 真实速度（m/s）
phi_true = [-45 15.5 20.89 ]';       % 方位角（度）
theta_true = [0.58 -15.55 5.55]';              % 俯仰角（度）
SNR = 15;                           % SNR(dB)
N_targets = length(R_true);         % 目标数量


% 预分配原始数据
Num_TX_A = 6
Num_TX_B = 8
group_idx = 2
tx_idx  = 1
N_subbands_B = 2
chirp_idx = 1
tx_idx = 1
t_abs = 1
phase_shift = 0.1
target_idx = 1
rx_signal_raw_A = zeros(Num_chirp_total/2, Samples_per_chirp, Num_TX_A);  % A chirp物理接收天线RX收到的数据立方体cube
rx_signal_raw_B = zeros(Num_chirp_total/2, Samples_per_chirp, Num_TX_B);  % B chirp物理接收天线RX收到的数据立方体cube
tx_signal = zeros(1, Samples_per_chirp); % 初始化发射信号
phase_shift_per_chirp(chirp_idx, tx_idx) = mod(2 * pi * group_idx * (tx_idx - 1) / N_subbands_B, 2*pi);
tx_signal = exp(1j * 2 * pi * (centerFreq * (t + t_abs) + 0.5 * Slope * t.^2) + 1j * phase_shift);
phase_if = 2 * pi * ((centerFreq - fd(target_idx)) * (t - tau(target_idx) + t_abs)) + pi * Slope * (t - tau(target_idx)).^2;
array_phase = (2 * pi / lambda) * (virt_pos_dy * sind(phi_true(target_idx)) * cosd(theta_true(target_idx)) + virt_pos_dz * sind(theta_true(target_idx))); 
% 回波信号合成
rx_signal = exp(1j * (phase_if - array_phase));


%% 3.0 距离维FFT和速度维FFT生成RDM
win_range = hamming(size(if_signals_Slowchirp, 2), 'periodic')'; 

if_signals_Slowchirp_windowed = if_signals_Slowchirp.* win_range; % 沿列（第二维）乘窗函数
if_signals_Fastchirp_windowed = if_signals_Fastchirp.* win_range; % 沿列（第二维）乘窗函数

% 3.1.3 距离维FFT（快时间）
range_fft_Slowchirp = fft(if_signals_Slowchirp_windowed, range_fft_bin, 2);
range_fft_Fastchirp = fft(if_signals_Fastchirp_windowed, range_fft_bin, 2);

% 3.2.1 生成第一维（慢时间）窗函数
win_velocity = hamming(size(if_signals_Slowchirp, 1), 'periodic'); 

% 3.2.2 对第一维加窗（依据if_group维数,自动扩展至三维数组的所有通道）
range_fft_Slowchirp_windowed = range_fft_Slowchirp.* win_velocity; 
range_fft_Fastchirp_windowed = range_fft_Fastchirp.* win_velocity; 
% 3.2.3 速度维FFT（慢时间）
velocity_fft_Slowchirp = fft(range_fft_Slowchirp_windowed, velocity_fft_bin, 1);
velocity_fft_Fastchirp = fft(range_fft_Fastchirp_windowed, velocity_fft_bin, 1);