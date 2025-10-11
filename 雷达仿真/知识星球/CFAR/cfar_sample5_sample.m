clear; close all; clc;
% 生成一个例子，然后用以下算法处理
% CA CFAR, OS CFAR, SOCA CFAR, GOCA CFAR, S-CFAR, Log CFAR
% 位置在50和55处有目标，50处信噪比10dB，55处信噪比15dB
% 前100单元噪声功率20dB，后100单元噪声功率30dB

num_cell = 200;
Pfa = 10^(-5);

signal1 = generateDataGaussianWhite(100, [50, 55], [35, 40], 20);
signal2 = generateDataGaussianWhite(100, [], [], 30);
signal = [signal1, signal2];
plot(1:num_cell, pow2db(signal), 'k-', 'linewidth', 0.5);
hold on;

% CA CFAR处理
[position, threshold, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
plot(start_cell:stop_cell, pow2db(threshold), 'linewidth', 1);
% OS CFAR处理
[position, threshold, start_cell, stop_cell] = oscfar(signal, Pfa, 10, 2, 15);
plot(start_cell:stop_cell, pow2db(threshold), 'linewidth', 1);
% SOCA CFAR处理
[position, threshold, start_cell, stop_cell] = socacfar(signal, Pfa, 10, 2);
plot(start_cell:stop_cell, pow2db(threshold), 'linewidth', 1);
% GOCA CFAR处理
[position, threshold, start_cell, stop_cell] = gocacfar(signal, Pfa, 10, 2);
plot(start_cell:stop_cell, pow2db(threshold), 'linewidth', 1);


grid on;
legend('信号', 'CA CFAR阈值', 'OS CFAR阈值', 'SOCA CFAR阈值', 'GOCA CFAR阈值');