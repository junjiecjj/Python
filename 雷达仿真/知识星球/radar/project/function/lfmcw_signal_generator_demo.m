clear; clc; close all
BW = 250e6; %带宽
Fc = 24e9; % 载波频率
T_chrip = 420e-6; %  chirp 持续时间
T_idle = 580e-6; % 两个chirp之间的间隔时间
Fs = 2.5e6;
numADC = 256; % # ADC采样点数/chrip
numChirps = 2; % # chrip/frame
numCPI = 1; % frame数量
axis_t = 0:1 / Fs:(T_chrip + T_idle) * numCPI * numChirps; % 时间轴
lambda = physconst('LightSpeed') / Fc;
d_rx = lambda / 2; % 接收天线之间的距离，单位米
d_tx = 4 * d_rx; % 发射天线之间的距离，单位米
numTx = 1; % 发射天线数量
numRx = 3; % 接收天线数量

% 目标初始化信息，每个目标用长度为6的列向量保存信息，前三位代表起始坐标，后三位代表速度。
target_info = [[10 * cosd(10), 10 * sind(10), 0, 0, 0, 0].', ...
                    [20 * cosd(-45), 20 * sind(-45), 0, 1 * cosd(-45), 1 * sind(-45), 0].'];
% 发射天线的位置，xyz坐标系
tx_pos = [zeros(1, numTx); linspace(-0.5, 0.5, numTx) * d_tx * (numTx - 1); zeros(1, numTx)];

% 接受天线的位置
rx_pos = [zeros(1, numRx); linspace(-0.5, 0.5, numRx) * d_rx * (numRx - 1); zeros(1, numRx)];

[channel_signal, ref_signal] = lfmcw_signal_generator(Fc, BW, T_chrip, T_idle, axis_t, tx_pos, rx_pos, target_info);

%% 绘制时域波形图

figure('Name', '单目标回波')
subplot(211);
plot(axis_t, real(ref_signal(1, :)));
title('目标1回波');
subplot(212);
plot(axis_t, real(ref_signal(2, :)));
title('目标2回波');
figure('Name', '通道1混频信号')
plot(axis_t, real(channel_signal{1, 1}));
title('通道1混频信号');
