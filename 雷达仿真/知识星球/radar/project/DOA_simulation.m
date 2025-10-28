clear; clc; close all
addpath '.\function'
bandWidth = 250e6; %带宽
fc = 24e9; % 载波频率
T_chrip = 420e-6; %  chirp 持续时间
T_idle = 580e-6; % 两个chirp之间的间隔时间
Fs = 2.5e6;
SNR = 10; % 噪声信号比

numADC = 256; % # ADC采样点数/chrip
numChirps = 32; % # chrip/frame
numCPI = 16; % frame数量
numTx = 1; % 发射天线数量
numRx = 8; % 接收天线数量
numChannel = numTx * numRx;

max_range = 255 * physconst('LightSpeed') / 2 / (bandWidth * numADC * (1 / Fs) * 4 / T_chrip);
max_velocity = (numChirps - 1) * physconst('LightSpeed') / (2 * fc * (T_chrip + T_idle) * numChirps);
axis_t = 0:1 / Fs:(T_chrip + T_idle) * numCPI * numChirps; % 时间轴
axis_range = (255:-1:0) * physconst('LightSpeed') / 2 / (bandWidth * numADC * (1 / Fs) * 4 / T_chrip); %输入信号混频后是负频率，因此坐标轴反向
axis_velocity = (floor(numChirps / 2):-1: - floor(numChirps / 2) + 1) / (numChirps - 1) * max_velocity;
axis_angle = (-90:1:90);

lambda = physconst('LightSpeed') / fc;
d_rx = lambda / 2; % 接收天线之间的距离，单位米
d_tx = 4 * d_rx; % 发射天线之间的距离，单位米

% 目标初始化信息，每个目标用长度为6的列向量保存信息，前三位代表起始坐标，后三位代表速度。
target_info = [[10 * cosd(10), 10 * sind(10), 0, 0, 0, 0].', ...
                    [20 * cosd(-45), 20 * sind(-45), 0, 1 * cosd(-45), 1 * sind(-45), 0].'];
% 发射天线的位置，xyz坐标系
tx_pos = [zeros(1, numTx); linspace(-0.5, 0.5, numTx) * d_tx * (numTx - 1); zeros(1, numTx)];

% 接受天线的位置
rx_pos = [zeros(1, numRx); linspace(-0.5, 0.5, numRx) * d_rx * (numRx - 1); zeros(1, numRx)];

%% 初始化matlab phased 工具箱 变量
phased_ula = phased.ULA('NumElements', numRx * numTx, 'ElementSpacing', d_rx);
ula_pos = getElementPosition(phased_ula);

%% 生成信号
[channel_signal, ref_signal] = lfmcw_signal_generator(fc, bandWidth, T_chrip, T_idle, axis_t, tx_pos, rx_pos, target_info);

%% 绘制时域波形图

% figure(1)
% subplot(211);
% plot(real(ref_signal(1, :)));
% subplot(212);
% plot(real(ref_signal(2, :)));
% figure(2)
% plot(real(channel_signal{1,1}));

%% 从ADC数据中抽取出radar_data_cube
% 从channel_signal中抽取出需要的数据组成radar_data_cube
% radar_data_cube 是一个四维数组,四个维度分别为: 帧编号 通道编号 chrip编号 采样点数

offset = 10;
start_index = (0:T_chrip + T_idle:max(axis_t));
start_index = start_index(1:end - 1);
start_index = round(start_index .* Fs + offset);
start_index = reshape(start_index, [numChirps, numCPI]);
radar_data_cube = zeros(numCPI, numel(channel_signal), length(start_index), numADC);
for i = 1:numTx
    for j = 1:numRx
        for k = 1:numCPI
            for l = 1:length(start_index)
                radar_data_cube(k, (i - 1) * numTx + j, l, :) = channel_signal{i, j}(start_index(l, k):4:start_index(l, k) + numADC * 4 - 1);
            end
        end
    end
end
clear channel_signal axis_t;
%% 添加噪声
k = mean(abs(radar_data_cube), 'all') / (10 ^ (SNR / 20));
radar_data_cube = radar_data_cube + k * (1j .* randn(size(radar_data_cube)) + 1 .* randn(size(radar_data_cube)));

%% 距离维FFT
rangeFFT = fft(radar_data_cube, 256, 4);

%% 绘制 RDM

% figure('Name', 'RDM')
% for i = 1:1
%     rdm = squeeze(radar_data_cube(1,i,:, :));
%     spec = fftshift(fft2(rdm, 32, 256), 1);
%     h = imagesc( axis_range,axis_velocity, abs(spec));
%     drawnow;
%     pause(0.1);
% end

%% 导向矢量矩阵 （MVDR和MUSIC通用）
steering_vectors = steering_vector(ula_pos, [axis_angle; zeros(size(axis_angle))], lambda);

%% 取出一帧数据
frame = squeeze(radar_data_cube(1, :, :, :));
x = squeeze(mean(frame(:, 1:1, :), 2)); %累计不同chrip的信号（高速运动的目标累积可能会带来角度模糊）
x = x - mean(x, 2); % 减去均值
Cxx = x * x';
%% Capon算法测角——matlab工具箱 phased.MVDRBeamformer

mvdrspatialspect = phased.MVDREstimator('SensorArray', phased_ula, ...
    'OperatingFrequency', fc, 'ScanAngles', axis_angle, ...
    'DOAOutputPort', true, 'NumSignals', 2);
[matlab_mvdr_spec, target_angs] = mvdrspatialspect(x.');

disp("Phased 工具箱MVDR算法检测到的目标角度:");
printVar(target_angs);

% figure("Name", 'MVDR 谱 (MATLAB函数)');
% helperPlotSpec(axis_angle, matlab_mvdr_spec, 'MVDR 谱 (MATLAB函数)');

%% Capon算法测角——自己实现

my_mvdr_spec = 1 ./ sum(steering_vectors' / Cxx .* steering_vectors.', 2);
my_mvdr_spec = sqrt(abs(my_mvdr_spec));
figure("Name", "Capon 算法测角")
hold on;
helperPlotSpec(axis_angle, matlab_mvdr_spec, 'MVDR 谱 (MATLAB函数)');
helperPlotSpec(axis_angle, my_mvdr_spec, 'MVDR 谱 (自己实现)', 'r*');
hold off;

%% MVDR 波束成型还原输入信号
[peaks, locs] = findpeaks(my_mvdr_spec, 'SortStr', 'descend');
peaks = peaks(1:2);
locs = locs(1:2);

strvec = steering_vectors(:, locs);

y = strvec' * x;

figure, plot(real(y(1, :)));
figure, plot(real(y(2, :)));

%% MUSIC算法 直接测角
% 系统的输入信号 $x(n)$

[Qn, P] = getNoiseSpace(Cxx);
my_music_spec = 1 ./ (sum(abs((steering_vectors' * Qn)) .^ 2, 2));
my_music_spec = sqrt(my_music_spec);

% figure("Name", "MUSIC 算法测角")
% helperPlotSpec(axis_angle, my_music_spec, 'MUSIC 谱 (自己实现)');

%% 使用matlab的musicdoa()函数测角
% [doas,spec,specang] = (covmat,nsig ,'ScanAngles',scanangle)
% covmat: 协方差矩阵
% nsig  : 信号个数
% scanangle : 扫描角度数组
% doas  : 目标角度
% spec  : MUSIC伪谱
% specang ：角度坐标轴

[doas, matlab_music_spec, specang] = musicdoa(Cxx, 2, 'ScanAngles', axis_angle, 'ElementSpacing', 0.5);
printVar("musicdoa()检测到的目标角度:", doas);

figure("Name", "MUSIC 算法测角")
hold on;
helperPlotSpec(axis_angle, my_music_spec, 'MUSIC 谱 (自己实现)');
helperPlotSpec(axis_angle, matlab_music_spec, 'MUSIC 谱 (Matlab 函数)', 'r*');
hold off;

%% MUSIC算法 距离维度分离后测角
frame = squeeze(radar_data_cube(1, :, :, :));
frame = fft(frame, 256, 3);
range_az_music = zeros(numADC, length(axis_angle));
for i = 1:numADC
    x = squeeze(frame(:, :, i));
    Cxx = x * x';
    [Qn, P] = getNoiseSpace(Cxx);
    range_az_music(i, :) = P ./ (sum(abs((steering_vectors' * Qn)) .^ 2, 2));
end

figure('Name', 'MUSIC 距离-角度谱')
helperPlotSpec2d(axis_angle, axis_range, range_az_music, 'Range-Angle Map');

%% Cpaon算法 距离维度分离后测角
frame = squeeze(radar_data_cube(1, :, :, :));
frame = fft(frame, 256, 3);
range_az_mvdr = zeros(numADC, length(axis_angle));
for i = 1:numADC
    x = squeeze(frame(:, :, i));
    Cxx = x * x';
    [Qn, P] = getNoiseSpace(Cxx);
    range_az_mvdr(i, :) = 1 ./ abs(sum(steering_vectors' / Cxx .* steering_vectors.', 2));
end

figure('Name', 'MUSIC 距离-角度谱')
helperPlotSpec2d(axis_angle, axis_range, range_az_mvdr, 'Capon');

%% MUSIC算法——分离噪声子空间
function [Un, P] = getNoiseSpace(Rxx)
    [V, D] = eig(Rxx); % Q: eigenvectors (columns), D: eigenvalues
    [D, I] = sort(diag(D), 'descend');
    V = V(:, I);
    th = mean(D);
    i = 1;
    while D(i) > th
        i = i + 1;
    end
    Un = V(:, i:end);
    P = sum(D(1:i - 1));
end
