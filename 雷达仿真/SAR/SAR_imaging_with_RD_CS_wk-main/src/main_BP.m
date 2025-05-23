%{
    本代码用于对雷达的回波数据，利用BP算法进行成像，利用电平饱和法以及直方图均衡法，
    提高成像质量。
    2025/3/18 21:36
    修复叠加部分斜距计算BUG，实现聚焦
    2025/3/27 15:40
%}
%% 数据读取
% 加载数据
clear;
clc;
close all;
echo1 = importdata('CDdata1.mat');
echo2 = importdata('CDdata2.mat');
% 将回波拼装在一起
echo = single([echo1;echo2]);
% echo = echo(1:512,:);
% 加载参数
para = importdata('CD_run_params.mat');
Fr = para.Fr;   % 距离向采样率
Fa = para.PRF;  % 方位向采样率
f0 = para.f0;   % 中心频率
Tr = para.Tr;   % 脉冲持续时间
R0 = para.R0;   % 最近点斜距
Kr = -para.Kr;   % 线性调频率
c = para.c;     % 光速
% 以下参数来自课本附录A
Vr = 7062;      % 等效雷达速度
Ka = 1733;      % 方位向调频率
f_nc = -6900;   % 多普勒中心频率
La = 20;        % 天线实孔径

% 星载SAR受限于波束宽度，一个方位向无法看到所有的栅格点
% 假如波宽较窄，距离向宽度较小，可直接视为矩形区域，使用矩形遮罩
% 是否使用矩形遮罩?
is_use_rect_mask = true;

% 如果你的电脑没有独显或者显存不够，请关闭该选项，实测至少需要6GB的显存
% 是否使用GPU加速？
is_use_gpu = true;

% 计算参数
lambda = c/f0;  % 波长
% 计算斜视角
theta_rc = asin(f_nc * lambda / 2 / Vr);
% 计算波束宽度
theta_bw = 0.886*lambda/La;

%% 图像填充
% 计算参数
[~,Nr] = size(echo);
% 距离向补零,方位向不处理
echo = padarray(echo,[0, round(Nr/3)]);


%% 轴产生
[Na,Nr] = size(echo);
% 距离向频率轴
fr_gap = Fr/Nr;
fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (0:Na-1)/Fa;    % 方位向时间轴
% 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis';

%% 第一步 距离压缩
% BP算法不用做方位向下变频
% 距离向傅里叶变换
echo_s1 = fft(echo,[],2);
% 距离向距离压缩滤波器
echo_d1_mf = exp(1i*pi/Kr.*fr_axis.^2);
% 距离向匹配滤波
echo_s2 = echo_s1 .* echo_d1_mf;
% 回到时域
echo_s2 = ifft(echo_s2,[],2);

%% 第二步 生成成像栅格
% 使用斜距栅格
x_range = R0 * cos(theta_rc) + linspace(0,(Nr-1)/Fr * c / 2,Nr); % 横向范围
% 由于波束宽度的存在，实际看到的比雷达方位向运动轨迹要多半个波束宽度角
y_offset = R0 * tan(theta_bw/2);                            % 纵向展宽量
y_range = linspace(-y_offset,(Na-1)/Fa*Vr+y_offset,Na);     % 纵向范围
[X, Y] = meshgrid(x_range, y_range);                        % 网格矩阵
echo_s4 = zeros(size(X)) + 1i .* zeros(size(X));            % 投影栅格矩阵

%% 使用GPU进行加速
% 生成投影栅格矩阵
if  is_use_gpu && canUseGPU
    echo_s2 = gpuArray(echo_s2);
    echo_s4 = gpuArray(echo_s4);
    X = gpuArray(X);
    Y = gpuArray(Y);
end

%% 第三步 方位向累加
% 计算梯形遮罩参数
X_min = min(x_range);
k1 = tan(theta_bw/2);               % 上边斜率
k2 = -tan(theta_bw/2);              % 下边斜率
b1 = X_min * k1;                    % 上下边截距
b2 = X_min * k2;
% 距离向插值
up_rat = 4;                         % 插值系数
up_Nr = Nr * up_rat;                % 插值后的点数
echo_s3 = interpft(echo_s2,up_Nr,2);
% 提前算好一些参数避免重复运算
X_2 = X.^2;                         % X^2
phase_const = 1j*4*pi/lambda;
scale_factor = 2*Fr*up_rat/c;
Y_const = Y + R0*tan(theta_rc);
% 方位向累加
h = waitbar(0,'BPA');
for i = 1:Na
    % 计算栅格点到雷达的距离
    R = sqrt(X_2 + (Y_const - ta_axis(i)*Vr).^2);
    % 将距离转换成时间并将时间归化到时域点数
    idx = round((R-R0) * scale_factor);
    % 防止越界
    idx_valid = idx >= 1 & idx <= up_Nr;
    idx(~idx_valid) = 1;
    % 提取回波
    foo = echo_s3(i,:);
    % 生成遮罩
    % 根据雷达方位向位置移动截距
    b11 = b1 + ta_axis(i)*Vr;
    b21 = b2 + ta_axis(i)*Vr;
    if is_use_rect_mask
        % 矩形遮罩
        mask = (Y < b11) & (Y > b21);
    else
        % 梯形遮罩
        mask = (Y < k1 * X + b11) & (Y > k2 * X + b21);
    end
    % 应用遮罩并累加
    echo_s4 = echo_s4 + (foo(idx) .* mask .* idx_valid) .* exp(phase_const*R);
    % 更新进度条
    waitbar(i/Na);
end
close(h);
echo_s5 = abs(echo_s4);
% 绘制直方图
figure;
histogram(echo_s5(:),100);
saturation = 1e3;
figure;
% 上下翻转
echo_s5 = flip(echo_s5,1);
% 饱和处理
echo_s5(echo_s5 > saturation) = saturation;
imagesc(x_range,y_range,echo_s5);
title('处理结果(BP算法)')
% 直方图均衡
echo_res = gather(echo_s5 ./ saturation);
echo_res = adapthisteq(echo_res,"ClipLimit",0.004,"Distribution","exponential","Alpha",0.5);
figure;
imshow(echo_res);
