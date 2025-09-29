% OFDM雷达-通信一体化系统仿真（含模糊函数分析）
clc; clear; close all;

%% ===================== 1. 系统参数配置 =====================
c0 = physconst('LightSpeed');    
fc = 30e9;                        % 载波频率30GHz
lambda = c0 / fc;                 % 波长=0.01m
N = 256;                          % 子载波数
M = 16;                           % 符号数
delta_f = 15e3 * 2^6;             % 子载波间隔=1.536MHz
T = 1 / delta_f;                  % 符号时长≈651ns
Tcp = T / 4;                      % CP时长≈163ns
Ts = T + Tcp;                     % 总符号周期≈814ns
CPsize = N / 4;                   % CP长度=64点
bitsPerSymbol = 2;              
qam = 2^(bitsPerSymbol);          % 4-QAM调制

%% ===================== 2. 发射机模块 =====================
data = randi([0 qam - 1], N, M);  % N子载波×M符号的基带数据
TxData = qammod(data, qam, 'UnitAveragePower', true);  % 4-QAM调制（归一化功率）

% OFDM调制（IFFT+CP）
TxSignal_IFTT = ifft(TxData, N);  % 时域信号（N点IFFT）
TxSignal_cp = [TxSignal_IFTT(N - CPsize + 1: N, :); TxSignal_IFTT];  % 添加CP
TxSignal_time = reshape(TxSignal_cp, [], 1);  % 一维时域发射信号

%% ===================== 3. 通信信道与接收 =====================
% 多径信道参数
PowerdB = [0 -8 -17 -21 -25];     % 信道抽头功率（dB）
Delay = [0 3 5 6 8];              % 抽头延迟（采样点）
Power = 10.^(PowerdB/10);         % 功率转线性值
Lch = Delay(end) + 1;             % 信道长度=9点
h = zeros(1, Lch);
h(Delay + 1) = (randn(1, length(PowerdB)) + 1j*randn(1, length(PowerdB))) .* sqrt(Power/2);  % 瑞利衰落信道

% 通信传输（卷积+加噪）
ComSNRdB = 15;
RxSignal_com = conv(TxSignal_time, h);  % 多径信道卷积
RxSignal_com = RxSignal_com(1:length(TxSignal_time));  % 截断至发射信号长度
RxSignal_com = awgn(RxSignal_com, ComSNRdB, 'measured');  % 加15dB噪声

% 通信接收（去CP+FFT+MMSE均衡+解调）
RxSignal_com_reshape = reshape(RxSignal_com, [N + CPsize, M]);  % 恢复为(320,16)矩阵
RxSignal_remove_cp = RxSignal_com_reshape(CPsize + 1:end, :);  % 去除CP，得到(256,16)频域信号
RxData_com = fft(RxSignal_remove_cp, N);  % FFT变换到频域

% MMSE均衡
H_channel = fft([h zeros(1, N - Lch)]).';  % 频域信道响应（256,1）
H_channel = repmat(H_channel, 1, M);       % 扩展到(256,16)匹配符号数
C = conj(H_channel) ./ (abs(H_channel).^2 + 10^(-ComSNRdB/10));  % MMSE均衡器
demodRxData = qamdemod(RxData_com .* C, qam, 'UnitAveragePower', true);  % 解调

% 误码统计
errorCount = sum(sum(de2bi(demodRxData, bitsPerSymbol) ~= de2bi(data, bitsPerSymbol)));
disp(['通信误码数: ', num2str(errorCount)]);

%% ===================== 4. 雷达信道与目标探测 =====================
target_pos = 30;                  % 目标真实距离=30m
target_speed = 20;                % 目标速度=20m/s
target_delay = 2 * target_pos / c0;  % 双程时延=2e-7s（30m距离对应往返时间）
target_dop = 2 * target_speed / lambda;  % 目标多普勒频移=4000Hz（2*20/0.01）
RadarSNRdB = 30;                  % 雷达信噪比
RadarSNR = 10^(RadarSNRdB/10);

% 生成正确的子载波和符号索引网格
[kSub, mSym] = meshgrid(0:M-1, 0:N-1);  % 索引从0开始

% 修正的相位偏移公式
phase_shift = -1j*2*pi * (...
    fc * target_delay + ...          % 载波相位偏移
    mSym * delta_f * target_delay - ...  % 子载波频偏引起的相位偏移
    kSub * Ts * target_dop);         % 多普勒引起的相位偏移

noise = sqrt(1/2)*(randn(N, M) + 1j*randn(N, M));  % 复高斯噪声
RxData_radar = sqrt(RadarSNR) * TxData .* exp(phase_shift) + noise;  % 雷达接收频域信号

% 雷达信号处理（FFT距离估计）
dividerArray = RxData_radar ./ TxData;  % 抵消发射信号（去调制）
NPer = 16 * N;                          % 补零点数=4096（提升距离分辨率）
range_fft = ifft(dividerArray, NPer, 1);  % 距离维FFT（沿子载波方向）
range_power = abs(range_fft);             % 距离维功率
mean_range_power = mean(range_power, 2);  % 沿符号方向平均（抑制噪声）
mean_range_power = mean_range_power / max(mean_range_power);  % 归一化
mean_range_power_dB = 10*log10(mean_range_power + eps);  % 转dB

% 距离轴计算公式
range_axis = (0:NPer-1) * c0 / (2 * NPer * delta_f);  % 正确的距离轴公式
[~, rangeEst_idx] = max(mean_range_power);  % 找到功率峰值索引
distanceE = range_axis(rangeEst_idx);       % 估计距离

% 输出准确的距离结果
disp(['目标估计距离: ', num2str(round(distanceE, 2)), ' m (真实距离: ', num2str(target_pos), ' m)']);

%% ===================== 5. 模糊函数计算 =====================
% 时延和多普勒范围设置
tau_max = 2 * target_delay;  % 时延范围±2倍目标时延
fd_max = 2 * target_dop;     % 多普勒范围±2倍目标多普勒
tau_points = 150;            % 时延点数
fd_points = 150;             % 多普勒点数

tau_range = linspace(-tau_max, tau_max, tau_points);
fd_range = linspace(-fd_max, fd_max, fd_points);

% 获取发射信号和采样时间
s_t = TxSignal_time;
t_seq = (0:length(s_t)-1) * (1/(N*delta_f));  % 采样时间序列

% 预分配模糊函数矩阵
ambiguity_func = zeros(length(tau_range), length(fd_range));

% 计算模糊函数
for i = 1:length(tau_range)
    tau = tau_range(i);
    
    % 时移信号（线性插值）
    t_shifted = t_seq - tau;
    idx = t_shifted >= 0 & t_shifted <= max(t_seq);
    s_tau = interp1(t_seq, s_t, t_shifted(idx), 'linear', 0);
    
    % 填充零以匹配原始长度
    s_tau_padded = zeros(size(s_t));
    s_tau_padded(idx) = s_tau;
    
    for j = 1:length(fd_range)
        fd = fd_range(j);
        
        % 应用多普勒频移
        doppler_shift = exp(1j*2*pi*fd*t_seq.');
        
        % 计算互相关
        ambiguity_func(i, j) = abs(s_t' * (s_tau_padded .* doppler_shift));
    end
end

% 归一化模糊函数
ambiguity_func = ambiguity_func / max(ambiguity_func(:));

% 提取距离切片（零多普勒）和速度切片（零时延）
[~, zero_dop_idx] = min(abs(fd_range));  % 找到零多普勒索引
[~, zero_tau_idx] = min(abs(tau_range)); % 找到零时延索引

range_slice = ambiguity_func(:, zero_dop_idx);  % 距离切片
doppler_slice = ambiguity_func(zero_tau_idx, :); % 速度切片

% 转换为dB
range_slice_dB = 10*log10(range_slice + eps);
doppler_slice_dB = 10*log10(doppler_slice + eps);

%% ===================== 6. 图像绘制 =====================
figure('Position', [100, 100, 1200, 800]);

% 子图1：发射信号时域（含CP）
subplot(2, 4, 1);
t_tx = (0:length(TxSignal_time)-1) * (1/(N*delta_f)) * 1e9;  % 时间轴（ns）
plot(t_tx, real(TxSignal_time), 'b-', 'LineWidth', 1);
hold on;
cp_end_idx = CPsize*M;  % CP总长度=64*16=1024点
plot(t_tx(1:cp_end_idx), real(TxSignal_time(1:cp_end_idx)), 'r--', 'LineWidth', 1);
xlabel('时间 (ns)'); ylabel('幅度'); title('OFDM发射信号时域（含CP）');
legend('数据部分', '循环前缀(CP)'); grid on; hold off;

% 子图2：发射信号频域
subplot(2, 4, 2);
freq_tx = (-N/2:N/2-1)*delta_f/1e6;  % 频率轴（MHz）
Tx_fft_shift = fftshift(abs(TxData(:,1)));  % 第一个符号的频域幅度
plot(freq_tx, 10*log10(Tx_fft_shift + eps), 'g-', 'LineWidth', 1);
xlabel('频率 (MHz)'); ylabel('功率 (dB)'); title('OFDM发射信号频域'); grid on;

% 子图3：通信信道冲击响应
subplot(2, 4, 3);
stem(0:Lch-1, abs(h), 'm-', 'MarkerFaceColor', 'm', 'LineWidth', 1);
xlabel('延迟 (采样点)'); ylabel('幅度'); title('通信信道冲击响应'); grid on;

% 子图4：雷达距离-功率谱
subplot(2, 4, 4);
plot(range_axis, mean_range_power_dB, 'b-', 'LineWidth', 1);
hold on;
plot([target_pos, target_pos], [min(mean_range_power_dB), 0], 'r--', 'LineWidth', 1.2);
plot([distanceE, distanceE], [min(mean_range_power_dB), 0], 'g:', 'LineWidth', 1.2);
xlabel('距离 (m)'); ylabel('归一化功率 (dB)'); title('雷达目标距离-功率谱');
legend('功率谱', '真实距离', '估计距离'); grid on; xlim([0 60]); hold off;

% 子图5：模糊函数3D图
subplot(2, 4, 5);
[X, Y] = meshgrid(fd_range/1000, tau_range*1e9);  % X:多普勒(kHz), Y:时延(ns)
surf(X, Y, ambiguity_func, 'EdgeColor', 'none');
xlabel('多普勒频移 (kHz)'); ylabel('时延 (ns)'); zlabel('归一化模糊度');
title('OFDM信号模糊函数（3D）'); colormap(jet); colorbar; view(45, 30);

% 子图6：模糊函数等高线
subplot(2, 4, 6);
contourf(X, Y, ambiguity_func, 30, 'LineColor', 'none');
hold on;
% 标记目标真实位置
target_tau_idx = find(abs(tau_range - target_delay) == min(abs(tau_range - target_delay)));
target_fd_idx = find(abs(fd_range - target_dop) == min(abs(fd_range - target_dop)));
plot(fd_range(target_fd_idx)/1000, tau_range(target_tau_idx)*1e9, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');
xlabel('多普勒频移 (kHz)'); ylabel('时延 (ns)'); title('模糊函数等高线');
colormap(jet); colorbar; grid on; hold off;

% 子图7：距离切片（零多普勒）
subplot(2, 4, 7);
plot(tau_range*1e9, range_slice_dB, 'b-', 'LineWidth', 1.5);
hold on;
plot([target_delay*1e9, target_delay*1e9], [min(range_slice_dB), 0], 'r--', 'LineWidth', 1.5);
xlabel('时延 (ns)'); ylabel('归一化幅度 (dB)'); title('距离切片（零多普勒）');
legend('距离响应', '目标位置'); grid on; hold off;

% 子图8：速度切片（零时延）
subplot(2, 4, 8);
plot(fd_range/1000, doppler_slice_dB, 'b-', 'LineWidth', 1.5);
hold on;
plot([target_dop/1000, target_dop/1000], [min(doppler_slice_dB), 0], 'r--', 'LineWidth', 1.5);
xlabel('多普勒频移 (kHz)'); ylabel('归一化幅度 (dB)'); title('速度切片（零时延）');
legend('速度响应', '目标位置'); grid on; hold off;

sgtitle('OFDM雷达-通信一体化系统仿真结果（含模糊函数分析）', 'FontSize', 16, 'FontWeight', 'bold');

%% ===================== 7. 单独绘制模糊函数详细图 =====================
figure('Position', [200, 200, 1000, 800]);

% 3D模糊函数
subplot(2, 2, 1);
surf(X, Y, ambiguity_func, 'EdgeColor', 'none');
xlabel('多普勒频移 (kHz)'); ylabel('时延 (ns)'); zlabel('归一化模糊度');
title('OFDM信号模糊函数（3D）'); colormap(jet); colorbar; view(45, 30);

% 等高线图
subplot(2, 2, 2);
contourf(X, Y, ambiguity_func, 30, 'LineColor', 'none');
hold on;
plot(fd_range(target_fd_idx)/1000, tau_range(target_tau_idx)*1e9, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('多普勒频移 (kHz)'); ylabel('时延 (ns)'); title('模糊函数等高线');
colormap(jet); colorbar; grid on; hold off;

% 距离切片
subplot(2, 2, 3);
plot(tau_range*1e9, range_slice_dB, 'b-', 'LineWidth', 2);
hold on;
plot([target_delay*1e9, target_delay*1e9], [min(range_slice_dB), 0], 'r--', 'LineWidth', 2);
xlabel('时延 (ns)'); ylabel('归一化幅度 (dB)'); title('距离切片（零多普勒）');
legend('距离响应', '目标位置'); grid on; hold off;

% 速度切片
subplot(2, 2, 4);
plot(fd_range/1000, doppler_slice_dB, 'b-', 'LineWidth', 2);
hold on;
plot([target_dop/1000, target_dop/1000], [min(doppler_slice_dB), 0], 'r--', 'LineWidth', 2);
xlabel('多普勒频移 (kHz)'); ylabel('归一化幅度 (dB)'); title('速度切片（零时延）');
legend('速度响应', '目标位置'); grid on; hold off;

sgtitle('OFDM信号模糊函数详细分析', 'FontSize', 16, 'FontWeight', 'bold');