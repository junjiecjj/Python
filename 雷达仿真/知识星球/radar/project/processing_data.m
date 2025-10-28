clear; clc; close all
addpath '.\function'

%% 加载数据
data = load('./data/RadarData_2024_05_31_09_34_52.mat');
[numFrame, numChannel, numChrip, numSampling] = size(data.frames);
if(data.data_type == '1dfft')
radar_data_cube = ifft(data.frames, numSampling, 4);
else
radar_data_cube = data.frames;
end

%% 参数设置
bandWidth = 1000e6; %带宽
fc = 24e9; % 载波频率
T_chrip = 420e-6; %  chirp 持续时间
T_idle = 2588e-6; % 两个chirp之间的间隔时间
T_nop = 118e-6;
Fs = 2.5e6/8;

numTx = 1; % 发射天线数量
numRx = 2; % 接收天线数量

lambda = physconst('LightSpeed') / fc;
d_rx = lambda / 2; % 接收天线之间的距离，单位米
d_tx = 34e-3; % 发射天线之间的距离，单位米

% 目标初始化信息，每个目标用长度为6的列向量保存信息，前三位代表起始坐标，后三位代表速度。
target_info = [[10 * cosd(10), 10 * sind(10), 0, 0, 0, 0].', ...
                    [20 * cosd(-45), 20 * sind(-45), 0, 1 * cosd(-45), 1 * sind(-45), 0].'];
% 发射天线的位置，xyz坐标系
tx_pos = [zeros(1, numTx); linspace(-0.5, 0.5, numTx) * d_tx * (numTx - 1); zeros(1, numTx)];

% 接受天线的位置
rx_pos = [zeros(1, numRx); linspace(-0.5, 0.5, numRx) * d_rx * (numRx - 1); zeros(1, numRx)];

%% 计算坐标轴
max_velocity = (numChrip - 1) * physconst('LightSpeed') / (2 * fc * (T_chrip + T_idle) * numChrip);
axis_t = 0:1 / Fs:(T_chrip + T_idle) * numFrame * numChrip; % 时间轴
axis_range = (0:numSampling - 1) * physconst('LightSpeed') / 2 / (bandWidth * numSampling * (1 / Fs) / T_chrip); %输入信号混频后是负频率，因此坐标轴反向
axis_velocity = -1 .* (floor(numChrip / 2):-1: - floor(numChrip / 2) + 1) / (numChrip - 1) * max_velocity;
axis_angle = (-90:1:90);
T_frame = (T_chrip+T_idle)*numChrip+T_nop  ;
%% 导向矢量矩阵 （MVDR和MUSIC通用）
steering_vectors = steering_vector(rx_pos, [axis_angle; zeros(size(axis_angle))], physconst('LightSpeed') / (fc+bandWidth/2));

%% 观察相干性

frame = squeeze(radar_data_cube(10, :, :, :));
figure()
title("观察相干性")
hold on;
for i = 1:6
    chrip = squeeze(frame(1, i*2, :));
    plot(real(chrip))
end
m = squeeze(mean(frame(1, :, :),2));
plot(real(m),"-*")
hold off;

%% 计算chrip全局均值
chrip_mean = mean(radar_data_cube, [1, 3]);
% radar_data_cube = radar_data_cube -chrip_mean;

%% 观察距离
figure('Name', 'RDM')
for i = 1:10:numFrame
    raw = squeeze(radar_data_cube(i, 1, :, :));
    raw = squeeze(mean(raw,1));
    range_domain = fft(raw);
    range_spec = abs(range_domain);
    plot(range_spec);
    drawnow;
    pause(0.1);
end

%% 观察RDM

figure('Name', 'RDM')
for i = 1:10:numFrame
    raw = squeeze(radar_data_cube(i, 1, :, :));
    rdm = fftshift(fft2(raw), 1);
    spec = abs(rdm);
    h = imagesc(axis_range, axis_velocity, abs(spec));
    drawnow;
    pause(0.1);
end

%% 绘制gif
% 初始化图形窗口
figure('Name', 'MUSIC 距离-角度谱');
set(gcf, 'Color', 'w');
set(gcf,'Position',[0 0 1600 800])
% 初始化存储图像数据的cell数组
images = {};

% 定义 GIF 文件名
gif_filename = 'mvdr_spectrum.gif';

% 预先设置统一的颜色映射和颜色范围
caxis_range = [0 1e7]; % 根据你的数据范围选择合适的颜色范围
[Theta, Rho] = meshgrid(axis_angle+90, axis_range);
[X, Y] = pol2cart(Theta / 180 * pi, Rho);


for f = 1:size(radar_data_cube, 1)
    frame = squeeze(radar_data_cube(f, :, :, :));
    frame = fft(frame, size(frame, 3), 3);
    range_az_mvdr = zeros(size(frame, 3), length(axis_angle));
    for i = 1:size(frame, 3)
        x = squeeze(frame(:, :, i));
        Cxx = x * x';
        range_az_mvdr(i, :) = 1 ./ abs(sum(steering_vectors' / Cxx .* steering_vectors.', 2));
    end

    % [~,I] = max(range_az_mvdr,[],"all");
    % ang = floor((I-1)/size(range_az_mvdr,1))+1;
    % r = I-(ang-1)*size(range_az_mvdr,1);
    % target_x = r

    % range_az_mvdr = 20*log10(range_az_mvdr);
    % range_az_mvdr = range_az_mvdr-min(range_az_mvdr,[],"all");

    % 绘制当前帧
    imagesc(axis_angle, axis_range, range_az_mvdr);
    surf(X, Y, range_az_mvdr, 'EdgeColor', 'none');
    % xlim([0, 10]);
    % ylim([-10, 10]);
    daspect([1, 1, 1]);
    view(2); % 俯视图
    colorbar;
    xlabel('X');
    ylabel('Y');
    title(['Time ', num2str((f - 1) * ((T_chrip + T_idle) * numChrip + T_nop))]);
    clim(caxis_range); % 设置颜色范围
    drawnow;

    % 捕获当前图像帧
    frame = getframe(gcf);
    img = frame2im(frame);
    [imind, cm] = rgb2ind(img, 256);

    % 将图像写入 GIF 文件
    if f == 1
        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', T_frame);
    else
        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', T_frame);
    end
end
