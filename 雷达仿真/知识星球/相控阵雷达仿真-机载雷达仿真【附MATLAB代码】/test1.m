%% part1
clc;clear;close all;

%% 参数设置
eps = 0.0001;
c = 3e8;

B = 1e6;                         % 带宽1MHz
tau = 100e-6;                    % 脉宽100μs
fs = 5 * B;                      % 采样率
Ts = 1 / fs;                     % 采样间隔
K = B / tau;                     % 调频斜率
N = round(tau * fs);             % 采样点数

%% T1
t = linspace(-tau/2, tau/2, N);  % 选取采样点，在-T/2与T/2间生成N个点

% 生成LFM信号lfm
lfm = exp(1j * pi * K * t.^2);

% 时域波形
figure;
subplot(2,1,1);
plot(t*1e6, real(lfm), 'b');
xlabel('时间 (\mus)');ylabel('幅度');title('LFM信号时域波形（实部）');
grid on;axis tight;
subplot(2,1,2);
plot(t*1e6, imag(lfm), 'r');
xlabel('时间 (\mus)');ylabel('幅度');title('LFM信号时域波形（虚部）');
grid on;axis tight;

% 频域波形
f = linspace(-fs/2, fs/2, N);
S = fftshift(fft(lfm));
figure;
plot(f/1e6, abs(S));
xlabel('频率 (MHz)');ylabel('幅度');title('LFM信号频域波形');
grid on;axis tight;

% 模糊函数
[afmag, delay, doppler] = ambgfun(lfm, fs, 10e3);
figure('Position', [100, 100, 700, 500]);
surf(delay*1e6, doppler/1e6, afmag);
shading flat;colormap jet;colorbar;
xlabel('时延\tau (\mus)');ylabel('多普勒频率F_D (MHz)');zlabel('幅度');
title('LFM信号模糊函数');
clim([0 1]); % 设置动态范围
% view(0, 90);   % 俯视视角展示刀刃特性
% 等高线图
figure('Position', [100, 100, 600, 500]);
ambgfun(lfm, fs, 1/tau);
xlabel('时延\tau (\mus)');ylabel('多普勒频率F_D (MHz)');title('LFM信号模糊函数等高线图');
xlim([-100,100]);ylim([-1,1]);

% % 另外的方法画模糊函数
% fd_max = B; % 一般设成带宽B
% tau_delay = linspace(-tau,tau,N); % 时延范围
% F_D = linspace(-fd_max, fd_max, N); % 多普勒频率范围
% chi = zeros(length(tau_delay), length(F_D)); % 初始化模糊函数
% % 计算模糊函数的值
% for i = 1:length(tau_delay)
%     for j = 1:length(F_D)
%         if abs(tau_delay(i)) < tau % 判断是否在tau内
%             chi(i, j) = ( 1 - abs( tau_delay(i) )/tau ) ...
%                 * abs( sin( pi*( F_D(j)*tau+B*tau_delay(i) )*( 1-abs( tau_delay(i) )/tau ) )...
%                 /( pi*( F_D(j)*tau+B*tau_delay(i) )*( 1-abs( tau_delay(i) )/tau ) )  );
%         end
%     end
% end
% % 三维图
% figure;
% surf(tau_delay*1e6, F_D/1e6, chi);
% xlabel('时延\tau (\mus)');ylabel('多普勒频率F_D (MHz)');zlabel('模糊函数幅度');
% title('LFM脉冲信号模糊函数');
% shading flat;colormap jet;colorbar;
% % 等高线图
% figure;
% contour(tau_delay*1e6, F_D/1e6, chi);
% xlabel('时延\tau (\mus)');ylabel('多普勒频率F_D (MHz)');title('LFM脉冲信号模糊函数等高线图');
% xlim([-100,100]);ylim([-1,1]);
% grid on;colorbar;

%% T2
R_t = 90e3; % 目标距离90km
v_t= 60;   % 目标速度60m/s

% 生成LFM信号St
t = 0:Ts:tau-Ts;
St = exp(1j * pi * K *t.^2);

td = 2 * R_t / c; % 目标时延 (s)
N_d = round(td * fs); % 目标时延采样点数
fc = 1e9;       % 假设载频1GHz
fd = 2 * v_t / (c / fc); % 多普勒频移 (Hz)

% 生成单目标回波信号
echo = [zeros(1,N_d), St.*exp(1j*2*pi*fd.*td)];
t1 =  ((1:length(echo)) - 1) * Ts * 1e3; % ms

% 绘制回波信号
figure;
subplot(2,1,1);
plot(t1, real(echo), 'b');
xlabel('时间 (ms)');ylabel('幅度');title('回波信号时域波形（实部）');
grid on;
subplot(2,1,2);
plot(t1, imag(echo), 'r')
xlabel('时间 (ms)');ylabel('幅度');title('回波信号时域波形（虚部）');
grid on;

% 匹配滤波
matched_filter = conj(fliplr(St));
mf = conv(echo, matched_filter);
mf = mf(length(St):end);
mf_abs = abs(mf);
t_mf = ((1:length(mf)) -1) * Ts * c / 2 / 1e3; % km

% 绘制结果
figure;
plot(t_mf, mf_abs);
xlabel('距离 (km)');ylabel('幅度');title('匹配滤波输出');
xlim([0 100]);
grid on;

%% T3

R_t = 90e3; % 目标距离90km
v_t= 60;   % 目标速度60m/s
t = 0:Ts:tau-Ts;
St = exp(1j * pi * K *t.^2); % 生成LFM信号St

PRF = 1e3;       % 脉冲重复频率1kHz
PRT = 1 / PRF;   % 脉冲重复间隔1ms
fc = 1e9;        % 载频1GHz
lambda = c / fc; % 波长
SNR_dB = -10;    % 信噪比-10dB
fd = 2 * v_t / lambda; % 多普勒频移 (Hz)
N_PRT = round(PRT * fs);          % 单个PRT的采样点数
num_pulses = 64;                  % 模拟的脉冲数量
N_st = length(St); % 单个脉冲内的采样点数

% 调用脉冲串接收信号函数
received = received_pulseburst(St, num_pulses, fd, PRF, td, N_PRT, N_d, N_st);

% 生成复高斯噪声
rx_power = mean(abs(received).^2);
SNR_linear = 10^(SNR_dB/10);
noise_power = rx_power / SNR_linear;
noise = sqrt(noise_power/2) * (randn(size(received)) + 1i*randn(size(received)));
rx_signal = received + noise;

% 绘制接收信号
figure('Name','脉冲串回波信号波形（实部）');
subplot(2,1,1);
plot(real(rx_signal));
xlabel('采样点');ylabel('幅度');title('接收信号实部');
xlim([0, min(10000, length(rx_signal))]);
subplot(2,1,2);
N_fft = 2^nextpow2(length(rx_signal));
f_fft = fs/2 * linspace(-1,1,N_fft);
spectrum = fftshift(fft(rx_signal, N_fft));
plot(f_fft/1e6, 20*log10(abs(spectrum)));
xlabel('频率 (MHz)');ylabel('幅度 (dB)');title('接收信号频谱');
xlim([-fs/2e6, fs/2e6]);
grid on;

% 匹配滤波与MTD处理
% 分帧处理，将接收信号分割为各脉冲
rx_pulses = reshape(rx_signal, N_PRT, num_pulses);
% 生成匹配滤波器
h_mf = conj(fliplr(St)).';
% 预分配存储空间
mf_output = zeros(N_PRT + N_st -1, num_pulses);
for i = 1:num_pulses
    pulse = rx_pulses(:, i);
    mf_output(:, i) = conv(pulse, h_mf, 'full');
end

% 距离轴校正（补偿匹配滤波引入的延迟）
range_bins_mf = ( (0:size(mf_output,1)-1) - (N_st-1) ) * (c/(2*fs));
valid_idx = range_bins_mf >= 0;  % 去除负距离
range_bins_mf = range_bins_mf(valid_idx) / 1e3;
mf_output = mf_output(valid_idx, :);

% 匹配滤波后波形
figure('Name','匹配滤波后的波形', 'Position', [50, 50, 700, 500]);
for i = 1:4
    subplot(4,1,i);plot(abs(mf_output(:,i)));
    xlabel('采样点');ylabel('幅度');
end

% MTD处理（多普勒FFT）
mtd = fftshift(fft(mf_output, [], 2), 2);
mtd_abs = abs(mtd);
mtd_db = 20*log10(mtd_abs / max(mtd_abs(:)) + eps);

% MTD后波形
figure('Name','MTD后的波形', 'Position', [50, 50, 700, 500]);
for i = 1:4
    subplot(4,1,i);plot(mtd_abs(:,i));
    xlabel('采样点');ylabel('幅度');
end

% 计算速度轴
doppler_bins = (-num_pulses/2:num_pulses/2-1) * PRF / num_pulses; % 多普勒频率（Hz）
speed_bins_mf = doppler_bins * lambda/2; % 速度轴（m/s）

% 找到幅度最大的点（即目标位置）
[~, idx] = max(mtd_abs(:)); 
[range_idx, doppler_idx] = ind2sub(size(mtd_abs), idx);
% 转换为实际物理量
detected_range_mf = range_bins_mf(range_idx);
detected_speed_mf = speed_bins_mf(doppler_idx);

figure('Name','MTD','Position', [100, 100, 700, 500]);
maxvalue = max(max(mtd_abs));
image(range_bins_mf,speed_bins_mf,255 * mtd_abs.' / maxvalue);
title('MTD');
xlim([80,100]);ylim([50,70]);
colormap jet;colorbar;
hold on;
% 标记真实目标的信息
plot(90, 60, 'go', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'g');
text_str = sprintf('(%.1f km, %.1f m/s)', 90, 60);
text(90, 60, text_str, 'FontSize', 12, ...
    'Color', 'g', 'HorizontalAlignment', 'center');
% 标记检测目标的信息
hold on;
plot(detected_range_mf, detected_speed_mf, ...
    'rp', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
text_str = sprintf('(%.1f km, %.1f m/s)', detected_range_mf, detected_speed_mf);
text(detected_range_mf, detected_speed_mf, ...
    text_str, 'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center');
legend('真实目标', '检测目标');

% 绘制三维图
[X, Y] = meshgrid(range_bins_mf, speed_bins_mf);
figure('Position', [50, 50, 800, 600]);
surf(X, Y, mtd_db.');
shading interp;colormap jet;colorbar;
xlabel('距离 (km)'); ylabel('速度 (m/s)'); zlabel('幅度 (dB)');
title('MTD输出"距离-速度-幅度"三维图');
% view(90, 0);
% view([0,90,0]);
grid on;
% 标记检测目标的信息
hold on;
plot3(detected_range_mf, detected_speed_mf, mtd_db(range_idx, doppler_idx), ...
    'rp', 'MarkerSize', 5, 'LineWidth', 2, 'MarkerFaceColor', 'r');
text_str = sprintf('(%.1f km, %.1f m/s)', detected_range_mf, detected_speed_mf);
text(detected_range_mf, detected_speed_mf, mtd_db(range_idx, doppler_idx)+3, ...
    text_str, 'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center');
legend('最终波形','检测目标');

%% T4

N_R = 16;           % 阵元数
d = lambda / 2;     % 阵元间距（m）
theta_target = 0;   % 目标方位角 (度)

% 阵列方向图生成
theta = -90:0.1:90; % 方位角范围（-90°到90°）
weights_uniform = ones(N_R,1);
weights_hamming = hamming(N_R); % Hamming窗加权

% 计算阵列方向图
array_pattern_u = zeros(1, length(theta));
array_pattern_w = zeros(1, length(theta));
for i = 1:length(theta)
    steering = exp(-1j*2*pi*d/lambda * sind(theta(i)) .* (0:N_R-1).'); % 每个θ的导向矢量
    array_pattern_u(i) = abs(weights_uniform' * steering); % 非Hamming窗加权求和
    array_pattern_w(i) = abs(weights_hamming' * steering); % Hamming窗加权求和
end
array_pattern_db = 20*log10(array_pattern_u / max(array_pattern_u));
array_pattern_w_db = 20*log10(array_pattern_w / max(array_pattern_w));

% 绘制方向图
figure('Position', [50, 50, 600, 500]);
plot(theta, array_pattern_db, 'b', 'LineWidth', 1.5);
hold on;
plot(theta, array_pattern_w_db, 'r', 'LineWidth', 1.5);
xlabel('方位角 (°)'); ylabel('增益 (dB)');title('16阵元均匀线阵方向图加窗前后对比');
legend('不加窗', 'Hamming窗');
xlim([-90 90]); ylim([-50 0]); grid on;

%% 模拟阵列发送和接收信号

% 生成阵列导向矢量
array_phase = exp(-1j * 2 * pi * d / lambda * sind(theta_target) * (0:N_R-1)).';

% 调用阵列接收信号函数
rx_array_signal = received_array(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st);

% 添加阵列噪声
noise = sqrt(noise_power/2) * (randn(size(rx_array_signal)) + 1i*randn(size(rx_array_signal)));
rx_array_signal = rx_array_signal + noise;

% DBF处理
% 计算加权导向矢量
weighted_steering = array_phase .* weights_hamming;  % 应用窗函数
% 波束形成权值归一化
w = weighted_steering / (array_phase' * weighted_steering);  % 保持主瓣增益
% 进行波束形成
rx_beamformed = rx_array_signal * conj(w);

% 分帧处理
rx_pulses = reshape(rx_beamformed, N_PRT, num_pulses);

% 匹配滤波
h_mf = conj(fliplr(St)).';
dbf_mf_output = zeros(N_PRT + N_st - 1, num_pulses);
for i = 1:num_pulses
    dbf_mf_output(:, i) = conv(rx_pulses(:,i), h_mf, 'full');
end

% 距离轴校正（补偿匹配滤波引入的延迟）
range_bins = ((0:size(dbf_mf_output,1)-1) - (N_st-1)) * (c/(2*fs));
valid_idx = range_bins >= 0;
range_bins = range_bins(valid_idx) / 1e3;
dbf_mf_output = dbf_mf_output(valid_idx, :);

% MTD处理
mtd_output = fftshift(fft(dbf_mf_output, [], 2), 2);
mtd_output_abs = abs(mtd_output);
mtd_output_db = 20*log10(mtd_output_abs / max(mtd_output_abs(:)) + eps);

% 计算速度轴
doppler_bins = (-num_pulses/2:num_pulses/2-1) * PRF / num_pulses; % 多普勒频率（Hz）
speed_bins = doppler_bins * lambda/2; % 速度轴（m/s）

% 绘制DBF-匹配滤波-MTD三维图
figure('Position', [50, 50, 800, 600]);
[X, Y] = meshgrid(range_bins, speed_bins);
surf(X, Y, mtd_output_db.');
shading interp;colormap jet;colorbar;
xlabel('距离 (km)'); ylabel('速度 (m/s)'); zlabel('幅度 (dB)');
title('DBF-脉冲压缩-MTD输出"距离-速度-幅度"三维图');
% view(90, 0);
% view([0,90,0]);
grid on;
% 标记检测目标的信息
% 找到幅度最大的点（即目标位置）
[~, idx] = max(mtd_output_abs(:)); 
[range_idx, doppler_idx] = ind2sub(size(mtd_output_abs), idx);
% 转换为实际物理量
detected_range = range_bins(range_idx);
detected_speed = speed_bins(doppler_idx);
hold on;
plot3(detected_range, detected_speed, mtd_output_db(range_idx, doppler_idx), ...
    'rp', 'MarkerSize', 5, 'LineWidth', 2, 'MarkerFaceColor', 'r');
text_str = sprintf('(%.1f km, %.1f m/s)', detected_range, detected_speed);
text(detected_range, detected_speed, mtd_output_db(range_idx, doppler_idx)+3, ...
    text_str, 'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center');
legend('最终波形','检测目标');



%% 生成阵列接收信号
function rx_array_signal = received_array(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)
    % 参数设置
    % St LFM发送信号
    % array_phase 阵列相位
    % N_R 阵元数
    % num_pulses 脉冲数
    % fd 多普勒频移
    % PRF 脉冲重复频率
    % td 目标延迟时间
    % N_PRT 单个PRT的采样点数
    % N_d 目标时延的采样点数
    % N_st 单个脉冲内的采样点数
    % 返回阵列接收信号 （脉冲数×单个PRT的采样点数，阵元数）

    % 初始化多通道接收信号
    rx_array_signal = zeros(num_pulses * N_PRT, N_R);
    for n = 0:num_pulses-1
        % 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = exp(1j * 2 * pi * fd * (n / PRF + td));
        % 生成阵列接收信号
        for k = 1:N_R
            % 每个阵元的相位补偿
            R_phase = array_phase(k) * doppler_phase;
            % 计算信号位置
            start_idx = n * N_PRT + N_d;
            end_idx = start_idx + N_st;
            % 截断处理
            if end_idx > num_pulses * N_PRT
                end_idx = num_pulses * N_PRT;
                valid_len = end_idx - start_idx;
                rx_array_signal(start_idx+1:start_idx+valid_len, k) = ...
                    rx_array_signal(start_idx+1:start_idx+valid_len, k) + ...
                    St(1:valid_len).' * R_phase;
            else
                rx_array_signal(start_idx+1:end_idx, k) = ...
                    rx_array_signal(start_idx+1:end_idx, k) + ...
                    St.' * R_phase;
            end
        end
    end
end

%% 生成脉冲串接收信号
function received = received_pulseburst(St, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)
    % 参数设置
    % St LFM发送信号
    % num_pulses 脉冲数
    % fd 多普勒频移
    % PRF 脉冲重复频率
    % td 目标延迟时间
    % N_PRT 单个PRT的采样点数
    % N_d 目标时延的采样点数
    % N_st 单个脉冲内的采样点数
    % 返回脉冲串接收信号 （脉冲数×单个PRT的采样点数，1）

    % 生成脉冲串接收信号
    received = zeros(num_pulses * N_PRT, 1);
    for n = 0:num_pulses-1
        % 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = exp(1j * 2 * pi * fd * (n / PRF + td));
        % 计算回波在接收窗口中的位置
        start_idx = n * N_PRT + N_d;
        end_idx = start_idx + N_st;

        % 截断处理防止越界
        if end_idx > num_pulses * N_PRT
            end_idx = num_pulses * N_PRT;
            valid_len = end_idx - start_idx;
            received(start_idx+1:end_idx) = received(start_idx+1:end_idx) + ...
                St(1:valid_len).' * doppler_phase;
        else
            received(start_idx+1:end_idx) = received(start_idx+1:end_idx) + ...
                St.' * doppler_phase;
        end
    end
end


