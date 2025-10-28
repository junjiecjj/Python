clear; clc; close all
%% 初始化参数，生成阵列信号

fs = 8000;
t = (0:1 / fs:1).';
x1 = cos(2 * pi * t * 300);
x2 = cos(2 * pi * t * 400);
% 生成均匀线阵
array = phased.ULA('NumElements', 10, 'ElementSpacing', 1);
array.Element.FrequencyRange = [100e6 300e6];
fc = 150.0e6;
x = collectPlaneWave(array, [x1 x2], [-10 0; 60 0]', fc);
noise = 0.1 * (randn(size(x)) + 1i * randn(size(x)));
signal = x + noise;
axis_angle = (-90:1:90);

%% Capon谱 phased.MVDREstimator
estimator = phased.MVDREstimator('SensorArray', array, 'ScanAngles', axis_angle, ...
    'OperatingFrequency', fc, 'DOAOutputPort', true, 'NumSignals', 2);
[y, doas] = estimator(signal);
doas = broadside2az(sort(doas), [20 -5])

%% Capon谱

addpath '.\function'
% 获取阵元坐标
ula_pos = getElementPosition(array);
% 计算导向矢量
steering_vectors = steering_vector(ula_pos, [axis_angle; zeros(size(axis_angle))], physconst('LightSpeed') / fc);
% 去直流 转置
x = transpose(signal - mean(signal, 1));
% 自协方差矩阵
Cxx = x * x';
my_mvdr_spec = 1 ./ sum(steering_vectors' / Cxx .* steering_vectors.', 2);
my_mvdr_spec = abs(my_mvdr_spec);
my_mvdr_spec = sqrt(my_mvdr_spec);

figure("Name", "Capon 算法测角")

hold on;
helperPlotSpec(axis_angle, y, 'MVDR 谱 (MATLAB函数)');
helperPlotSpec(axis_angle, my_mvdr_spec, 'MVDR 谱 (自己实现)', 'r*');
hold off;
