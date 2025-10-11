%% part2
clc;clear;close all;

%% 参数设定

c = 3e8;
eps = 0.0001;

% 雷达参数
N_R = 16;               % 阵元数
P_t = 2e3;              % 单阵元功率(W)
H_c = 5e3;              % 载机高度(m)
v_c = 150;              % 载机速度(m/s)
fc = 1e9;               % 载频(Hz)
lambda = c/fc;          % 波长(m)
B = 1e6;                % 带宽(Hz)
T_p = 100e-6;           % 脉宽(s)
PRF = 1e3;              % 脉冲重复频率(Hz)
CPI = 256;              % 积累脉冲数(CPI内脉冲数)
d = lambda/2;           % 阵元间距(m)
Ls_dB = 10;             % 接收机损耗(dB)
Ls = 10^(Ls_dB / 10);   % 转化为线性值
F_dB = 5;               % 噪声系数(dB)
F = 10^(F_dB / 10);     % 转化为线性值


% 目标参数
R_t = 90e3;        % 目标距离(m)
RCS_t = 5;         % 目标RCS(m²)
v_t = 60;          % 目标径向速度(m/s)

% 杂波参数
sigma0 = 0.01;          % 杂波后向散射系数
N_bin = 101;            % 杂波块个数
T0 = 290;               % 标准温度(K)
k_B = 1.38e-23;         % 玻尔兹曼常数

% 仿真参数
k_sam = 20;            % 样本个数（杂波距离环个数）
azimuth_target = 0;     % 目标方位（°）
azimuth_array = 90;     % 阵列与载机飞行方向夹角（°）

%% T5

% 获取杂波数据
[x_clutter, ~, ~, ~] = ClutterGen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, CPI, N_R, d, lambda, PRF, B, k_sam, sigma0, P_t, Ls);

% % 空时二维FFT
N_fft_s = 512; N_fft_d = 512;
% % 选择第一个距离环的数据
clutter_data = reshape(x_clutter(:,1), N_R, CPI);
st_spectrum = fftshift(fft2(clutter_data, N_fft_s, N_fft_d));
st_spectrum_db = 20 * log10(abs(st_spectrum.') + eps);

% 生成频率轴
fs_axis = linspace(-0.5, 0.5, N_fft_s);       % 归一化空间频率 (d/λ)
fd_axis = linspace(-0.5, 0.5, N_fft_d);       % 归一化多普勒频率 (Hz)

% 绘制三维空时谱
figure('Name','正侧视单距离环杂波空时谱', 'Position', [100, 100, 800, 500]);
surf(fs_axis, fd_axis, st_spectrum_db, 'EdgeColor', 'none', 'DisplayName','杂波脊');
xlabel('归一化空间频率 (d/\lambda)'); ylabel('归一化多普勒频率F_d (Hz)'); zlabel('幅度 (dB)');
% xlim([-0.5,0.5]);ylim([-0.5,0.5]);
title('正侧视单距离环杂波空时谱'); shading interp;colormap jet; colorbar;
axis tight;grid on;

% 理论杂波脊线叠加
hold on;
slope = (2 * v_c) / (d * PRF);% 理论斜率
fs_theo = linspace(-0.5, 0.5, 100);
fd_theo = slope * fs_theo ;
max_db = max(st_spectrum_db(:)); % 获取频谱最大幅度
plot3(fs_theo, fd_theo, max_db * ones(size(fs_theo)), 'r--', 'LineWidth', 2, 'DisplayName','理论脊');
xlim([-0.5,0.5]);ylim([-0.5,0.5]);
% legend('杂波谱', '理论杂波脊');
legend show;
hold off;

%% T6

% 信号波形生成
fs = 2 * B;
Ts = 1 / fs;
td = 2 * R_t / c; % 目标时延 (s)
N_d = round(td * fs); % 目标时延采样点数
fd = 2 * v_t / lambda; % 目标多普勒频率
K = B/T_p;                          % 调频斜率
t_chirp = linspace(0, T_p-Ts, T_p*fs); 
St = exp(1j*pi*K*t_chirp.^2);       % LFM信号
N_st = length(St);                  % 单个脉冲采样点数
N_PRT = round(1/PRF * fs);          % 单个PRT采样点数
num_pulses = 256;                   % 脉冲数

% 阵列接收信号生成（含目标、杂波、噪声）
array_phase = exp(-1j * 2 * pi * d / lambda * sind(azimuth_target) * (0:N_R-1)).'; % 导向矢量

% 生成目标回波 (调用阵列接收信号函数)
rx_target = rx_array_airborneradar(St, array_phase, N_R, num_pulses,  fd, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda, Ls);

% 生成杂波
clutter = ClutterGen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, num_pulses*N_PRT, N_R, d, lambda, PRF, B, k_sam, sigma0, P_t, Ls);
rx_clutter = reshape(clutter(:,1), num_pulses*N_PRT, N_R);

% 生成噪声
noise_power = F * k_B * T0 * B;
noise = sqrt(noise_power/2)*(randn(size(rx_target)) + 1j*randn(size(rx_target)));

% 合成接收信号
rx_array_signal = rx_target + rx_clutter + noise; % [N_PRT*num_pulses, N_R]

% DBF处理
weights_hamming = hamming(N_R);      % 汉明窗
steering_vector = array_phase .* weights_hamming; 
w = steering_vector / (array_phase' * steering_vector); % 归一化权值
rx_beamformed = rx_array_signal * conj(w); % 波束形成

% 分帧处理
rx_pulses = reshape(rx_beamformed, N_PRT, num_pulses);

% 脉冲压缩
% 雷达发射LFM信号后接收信号的功率（雷达方程计算）
Pr = (P_t * lambda^2 * RCS_t) / ((4*pi)^3 * R_t^4 * Ls);
Ar = sqrt(Pr);
h_mf = conj(fliplr(Ar * St)).'; % 匹配滤波器
dbf_mf_output = zeros(N_PRT + N_st - 1, num_pulses);
for i = 1:num_pulses
    dbf_mf_output(:,i) = conv(rx_pulses(:,i), h_mf, "full"); 
end

% 距离轴校正（补偿匹配滤波引入的延迟）
range_axis = ((0:size(dbf_mf_output,1)-1) - (N_st-1)) * c/(2*fs);
valid_idx = range_axis >= 0;
range_axis = range_axis(valid_idx) / 1e3; % 转换为km
dbf_mf_output = dbf_mf_output(valid_idx,:);

% MTD处理（多普勒FFT）
mtd_output = fftshift(fft(dbf_mf_output, [], 2), 2);

mtd_output_abs = abs(mtd_output);
mtd_output_db = 20*log10(mtd_output_abs / max(mtd_output_abs(:)) + eps);

% 速度轴计算
doppler_axis = (-num_pulses/2:num_pulses/2-1) * PRF / num_pulses; 
speed_axis = doppler_axis * lambda/2;

% 三维可视化
figure('Name','机载雷达检测目标接收信号三维图', 'Position', [100, 100, 800, 500]);
[X,Y] = meshgrid(range_axis, speed_axis);
surf(X, Y, mtd_output_db.', 'EdgeColor','none');
shading interp; colormap jet; colorbar;
xlabel('距离 (km)'); ylabel('速度 (m/s)'); zlabel('幅度 (dB)');
title('距离-速度-幅度三维图');

% 标记目标
hold on;
[~, idx] = max(mtd_output_db(:));
[row, col] = ind2sub(size(mtd_output_db), idx);
detected_range = range_axis(row);
detected_speed = speed_axis(col);
plot3(detected_range, detected_speed, mtd_output_db(row,col), ...
    'rp', 'MarkerSize', 5, 'LineWidth', 2, 'MarkerFaceColor', 'r');
text_str = sprintf('(%.1f km, %.1f m/s)', detected_range, detected_speed);
text(detected_range, detected_speed, mtd_output_db(row,col)+3,...
    text_str, 'FontSize', 12,  'Color','r', 'HorizontalAlignment', 'center');
legend('最终波形','检测目标');

%% CFAR检测（二维CA-CFAR）

% CFAR参数设置
guard = [5, 5];     % 距离/速度保护单元
train = [10, 10];   % 距离/速度参考单元
P_fa = 1e-6;        % 虚警概率

% 执行CFAR检测
[detection_map, target_info] = cfar_2d(mtd_output_db, guard, train, P_fa,...
                                      range_axis, speed_axis, R_t, v_t);

% 打印检测结果
fprintf('===== 目标检测结果 =====\n');
fprintf('真实目标位置: %.2f km, %.2f m/s\n', target_info.TrueRange, target_info.TrueSpeed);
fprintf('检测目标位置: %.2f km, %.2f m/s\n', target_info.DetectedRange, target_info.DetectedSpeed);
fprintf('距离相对误差: %.2f%%\n', target_info.RangeError);
fprintf('速度相对误差: %.2f%%\n', target_info.SpeedError);

% 绘制检测结果
figure('Position',[100 100 800 500]);
imagesc(range_axis, speed_axis, detection_map.');
xlabel('距离 (km)'); ylabel('速度 (m/s)'); title('CFAR检测结果');
colormap jet; colorbar;
hold on;

% 标记真实目标位置
plot(target_info.TrueRange, target_info.TrueSpeed, 'go',...
    'MarkerSize',10,'LineWidth',2,'DisplayName','真实目标');
text_str = sprintf('(%.2f km, %.2f m/s)', target_info.TrueRange, target_info.TrueSpeed);
text(target_info.TrueRange, target_info.TrueSpeed+3, ...
    text_str, 'FontSize', 12, 'Color', 'g', 'HorizontalAlignment', 'center');

% 标记检测目标位置
if ~isnan(target_info.DetectedRange)
    plot(target_info.DetectedRange, target_info.DetectedSpeed, 'rp',...
        'MarkerSize',12,'LineWidth',2,'DisplayName','检测目标');
    text_str = sprintf('(%.2f km, %.2f m/s)', target_info.DetectedRange, target_info.DetectedSpeed);
    text(target_info.DetectedRange, target_info.DetectedSpeed-3, ...
    text_str, 'FontSize', 12, 'Color', 'r', 'HorizontalAlignment', 'center');
    legend show;
end

%% T7

% 参数设置
v_max = PRF * lambda / 4; % 归一化多普勒频率±0.5对应的最大速度
v_values = linspace(-v_max, v_max, 31); % 生成速度范围(-v_max到v_max,共31个点)
SNCR_dB = zeros(size(v_values)); % 存储各速度对应的SCNR

% 背景功率计算 (无目标时)
rx_background = rx_clutter + noise;
rx_beamformed_bg = rx_background * conj(w);
rx_pulses_bg = reshape(rx_beamformed_bg, N_PRT, num_pulses);

% 脉冲压缩和MTD处理
dbf_mf_output_bg = zeros(N_PRT + N_st - 1, num_pulses);
for i = 1:num_pulses
    dbf_mf_output_bg(:,i) = conv(rx_pulses_bg(:,i), h_mf, "full"); 
end

% 距离轴校正（补偿匹配滤波引入的延迟）
range_axis_bg = ((0:size(dbf_mf_output_bg,1)-1) - (N_st-1)) * c/(2*fs);
valid_idx_bg = range_axis_bg >= 0;
range_axis_bg = range_axis_bg(valid_idx_bg) / 1e3; % 转换为km
dbf_mf_output_bg = dbf_mf_output_bg(valid_idx_bg,:);

% MTD处理（多普勒FFT）
mtd_output_bg = fftshift(fft(dbf_mf_output_bg, [], 2), 2);
mtd_output_bg_abs = abs(mtd_output_bg);
mtd_power_bg = mean(mtd_output_bg_abs(:).^2); % 背景平均功率

% 遍历目标速度计算SCNR
for k = 1:length(v_values)
    v_k = v_values(k);
    fd_k = 2 * v_k / lambda; % 当前速度对应的多普勒频率
    
    % 生成目标信号
    rx_target_k = rx_array_airborneradar(St, array_phase, N_R, num_pulses, ...
        fd_k, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda, Ls);
    
    % 合成接收信号
    rx_array_signal_k = rx_target_k + rx_background;
    
    % 处理链
    rx_beamformed_k = rx_array_signal_k * conj(w);
    rx_pulses_k = reshape(rx_beamformed_k, N_PRT, num_pulses);
    
    % 脉冲压缩
    dbf_mf_output_k = zeros(N_PRT + N_st -1, num_pulses);
    for i = 1:num_pulses
        dbf_mf_output_k(:,i) = conv(rx_pulses_k(:,i), h_mf, "full"); 
    end
    dbf_mf_output_k = dbf_mf_output_k(valid_idx_bg,:);
    
    % MTD处理
    mtd_output_k = fftshift(fft(dbf_mf_output_k, [], 2), 2);
    mtd_output_k_abs = abs(mtd_output_k);
    mtd_power_k = mtd_output_k_abs.^2;
    
    % 定位目标单元
    [~, range_idx] = min(abs(range_axis_bg * 1e3 - R_t));
    [~, doppler_idx] = min(abs(speed_axis - v_k));
    
    % 计算SCNR
    signal_power = mtd_power_k(range_idx, doppler_idx);
    SNCR_dB(k) = 10*log10(signal_power / mtd_power_bg);
end

% 绘制SCNR曲线
figure('Position',[100 100 800 500]);
plot(v_values, SNCR_dB, 'LineWidth',1.5);
xlabel('目标速度 (m/s)'); ylabel('SCNR (dB)');
title('目标SCNR随速度变化曲线');
grid on;
hold on;

% 标记杂波脊位置
v_clutter_ridge = v_c * sind(azimuth_target); % 正侧视杂波速度
plot([v_clutter_ridge, v_clutter_ridge], ylim, 'r--', 'LineWidth',1.2);
legend('SCNR曲线', '杂波脊位置');


%% 二维CA-CFAR检测函数
function [detection_map, target_info] = cfar_2d(input_data, guard_win, train_win, P_fa, range_bins, speed_bins, R_true, v_true)
    % input_data: 输入数据矩阵（距离门×速度门）
    % guard_win: 保护单元 [距离保护, 多普勒保护]
    % train_win: 参考单元 [距离参考, 多普勒参考]
    % P_fa: 虚警概率
    % range_bins: 距离轴向量
    % speed_bins: 速度轴向量
    % R_true: 真实目标距离
    % v_true: 真实目标速度

    [num_range, num_doppler] = size(input_data);
    detection_map = zeros(num_range, num_doppler);

    % 计算阈值因子
    num_ref_cells = (2*train_win(1)+2*guard_win(1)+1)*(2*train_win(2)+2*guard_win(2)+1) - ...
                   (2*guard_win(1)+1)*(2*guard_win(2)+1);
    alpha = num_ref_cells*(P_fa^(-1/num_ref_cells) - 1);

    % 滑动窗口检测
    for range_idx = 1+train_win(1)+guard_win(1) : num_range-train_win(1)-guard_win(1)
        for doppler_idx = 1+train_win(2)+guard_win(2) : num_doppler-train_win(2)-guard_win(2)
            % 定义检测区域
            range_win = range_idx-train_win(1)-guard_win(1):range_idx+train_win(1)+guard_win(1);
            doppler_win = doppler_idx-train_win(2)-guard_win(2):doppler_idx+train_win(2)+guard_win(2);

            % 提取参考单元
            ref_cells = input_data(range_win, doppler_win);

            % 去除保护单元
            ref_cells(train_win(1)+1:end-train_win(1), train_win(2)+1:end-train_win(2)) = NaN;

            % 计算噪声基底
            noise_level = mean(ref_cells(:), "omitmissing");
            threshold = alpha * noise_level;

            % 检测判决
            if input_data(range_idx, doppler_idx) > threshold
                detection_map(range_idx, doppler_idx) = 1;
            end
        end
    end

    % 匹配真实目标
    [~, true_range_idx] = min(abs(range_bins - R_true/1e3));
    [~, true_speed_idx] = min(abs(speed_bins - v_true));

    % 判断真实目标是否被检测到
    is_detected = false;
    if detection_map(true_range_idx, true_speed_idx) == 1
        detected_range = range_bins(true_range_idx);
        detected_speed = speed_bins(true_speed_idx);
        is_detected = true;
    end

    if ~is_detected
        [detected_ranges, detected_speeds] = find(detection_map == 1);
        min_dist = inf;
        detected_range = NaN;
        detected_speed = NaN;
        for k = 1:length(detected_ranges)
            current_dist = sqrt(...
                (range_bins(detected_ranges(k)) - R_true/1e3)^2 + ...
                (speed_bins(detected_speeds(k)) - v_true)^2);
            if current_dist < min_dist
                min_dist = current_dist;
                detected_range = range_bins(detected_ranges(k));
                detected_speed = speed_bins(detected_speeds(k));
            end
        end
    end

    % ==== 错误处理：未检测到目标时给出警告 ====
    if isnan(detected_range)
        warning('未检测到真实目标，请调整CFAR参数!');
    end

    % 计算误差
    if ~isnan(detected_range)
        range_error = abs(detected_range - R_true/1e3)/(R_true/1e3)*100;
        speed_error = abs(detected_speed - v_true)/abs(v_true)*100;
    else
        range_error = NaN;
        speed_error = NaN;
    end

    % 输出结果
    target_info = struct(...
        'TrueRange', R_true/1e3, ...
        'TrueSpeed', v_true, ...
        'DetectedRange', detected_range, ...
        'DetectedSpeed', detected_speed, ...
        'RangeError', range_error, ...
        'SpeedError', speed_error);
end


%% ClutterGen.m 杂波建模函数
function [x_all, f_s, f_d, azimuth_c] = ClutterGen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, CPI_c, N_R, d, lambda, PRF, B, k_sam, sigma0, P_t, Ls)
    % 载机高度H_c（m），目标距离R_c（m），载机速度v_c（m/s），
    % H_c 载机高度(m)
    % R_t 目标距离(m)
    % v_c 载机速度(m/s)
    % azimuth_target 目标方位（°）
    % azimuth_array 阵列与载机飞行方向夹角（°）
    % N_bin 杂波块个数
    % CPI_c CPI内脉冲数
    % N_R 阵元数
    % d 阵元间隔（m）
    % lambda 波长(m)
    % PRF 脉冲重复频率(Hz)
    % B 带宽(Hz)
    % k_sam 样本个数（杂波距离环个数）
    % sigma0 杂波后向散射系数
    % P_t 发射功率 (W)
    % Ls 接收机损耗
    % 函数返回杂波信号x_all （N_R×CPI，k_sam+1）

    % 常数设置
    c = 3e8;            % 光速 (m/s)
    H = H_c;      % 载机高度（m）
    v = v_c;            % 载机速度（m/s）
    L = CPI_c;          % CPI内脉冲数
    delta_R = c / (2 * B); % 距离环间隔

    % % 添加噪声功率计算
    % k_B = 1.38e-23; % 玻尔兹曼常数
    % T0 = 290;       % 噪声温度
    % P_noise = k_B * T0 * B; % 噪声功率
    % CNR_dB = 40;
    % CNR_linear = 10^(CNR_dB/10);        % 1e4

    %--------------计算待测距离环和参考距离环的俯仰角----------------
    R = R_t;                      % 目标距离（m）
    R_all = R + delta_R * (-k_sam/2 : k_sam/2); % 所有距离环距载机的距离
    phi_all = asin(H ./ R_all);          % 所有距离环俯仰角

    % 杂波块数目和方向角设置
    azimuth_c = linspace(-pi/2, pi/2, N_bin); % 杂波块方位角/正侧视[-90,90]/正前视[0,180]
    theta_rel = azimuth_c - deg2rad(azimuth_array); % 考虑阵列方向
    d_theta = pi / (N_bin - 1);     % 方位角间隔

    %--------------各距离环杂波块的空时频率------------
    f_s = (d/lambda) * cos(phi_all)' * sin([linspace(-pi/2, 0, (N_bin-1)/2), 0, linspace(0, pi/2, (N_bin-1)/2)]); % 各距离环每个杂波块的空间频率
    f_d = (2*v/lambda) * cos(phi_all)' * sin([linspace(-pi/2, 0, (N_bin-1)/2), 0, linspace(0, pi/2, (N_bin-1)/2)]) / PRF; % 各距离环每个杂波块的多普勒频率

    Amplitude_all = zeros(k_sam+1,N_bin); % 各距离环各杂波块的复幅度
    x_all = zeros(N_R*L,k_sam+1);  % 所有距离环的杂波回波数据
    for ring_num = 1:length(R_all)
        R_ring = R_all(ring_num);
        phi = phi_all(ring_num);
        % 地面投影距离
        % R_ground = sqrt(R_ring^2 - H^2);
        R_ground = R_ring * cos(phi);

        %---------------计算各杂波块CNR和幅度---------------------
        area_patch = delta_R * R_ground * d_theta;
        RCS_patch = sigma0 * area_patch;

        % 雷达方程或者固定杂噪比计算幅度（天线方向图增益为1）
        Pr = (P_t * N_R^2 * lambda^2 * RCS_patch) / ((4*pi)^3 * R_ring^4 * Ls); % CNR×P_noise为杂波功率
        % Pr = CNR_linear * P_noise; % CNR×P_noise为杂波功率

        Amplitude_all(ring_num,:) = sqrt(Pr); % 根据雷达方程或预设杂噪比计算该距离环每个杂波块的复幅度

        % 空时导向矢量
        a_s = exp(1j*2*pi*(0:N_R-1)' * f_s(ring_num,:));
        a_t = exp(1j*2*pi*(0:L-1)' * f_d(ring_num,:));

        %-----------------------回波数据-----------------------
        for clutterpatch_num = 1:N_bin
            x_all(:,ring_num) = x_all(:,ring_num) + Amplitude_all(ring_num,clutterpatch_num) * kron(a_t(:,clutterpatch_num),a_s(:,clutterpatch_num));
            %该距离环各杂波块回波数据叠加
        end
    end
end


%% 生成机载雷达均匀线阵阵列接收信号
function rx_array_signal = rx_array_airborneradar(St, array_phase, N_R, num_pulses, ...
    fd, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda, Ls)
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
    % P_t 单阵元功率(W)
    % RCS_t 目标RCS(m²)
    % R_t 目标距离(m)
    % lambda 波长(m)
    % Ls 接收机损耗
    % 返回阵列接收信号 （脉冲数×单个PRT的采样点数，阵元数）

    % 计算目标接收功率（单阵元雷达方程）
    Pr = (P_t * lambda^2 * RCS_t) / ((4*pi)^3 * R_t^4 * Ls);
    A_t = sqrt(Pr);  % 目标信号幅度

    % % 生成幅度加权的发射信号
    % St_scaled = A_t * St;

    % 初始化多通道接收信号
    rx_array_signal = zeros(num_pulses * N_PRT, N_R) * A_t;
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



