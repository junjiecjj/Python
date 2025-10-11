% Simulationg for several types of cfar algorithms.

clear; close all; clc

% ======Generate data1=====================================================
% data1:
%   # of units: 200; noise power: 20 dB
%   target position: [50]; echo wave power: [35]dB
%   probability of false alarm: 1e-3
num_unit = 200;
pos_target = 50;
echo_power_dB = 35;
noise_power_dB = 20;
Pfa = 1e-3;
signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                   echo_power_dB, noise_power_dB);
signal_dB = 10 .* log10(signal);

% data2:
%   # of units: 200; noise power: 20 dB
%   target position: [50, 58]; echo wave power: [35, 40]dB
%   probability of false alarm: 1e-3
num_unit2 = 200;
pos_target2 = [50; 58];
echo_power_dB2 = [35; 40];
noise_power_dB2 = 20;
Pfa2 = 1e-3;
signal2 = generateDataGaussianWhite(num_unit2, pos_target2, ...
                                   echo_power_dB2, noise_power_dB2);
signal2_dB = 10 .* log10(signal2);

% data3:
%   # of units: 200; noise power: 20 dB
%   target position: [50, 51, 52]; echo wave power: [35, 35, 35]dB
%   probability of false alarm: 1e-3
num_unit3 = 200;
pos_target3 = [50; 51; 52];
echo_power_dB3 = [35; 35; 35];
noise_power_dB3 = 20;
Pfa3 = 1e-3;
signal3 = generateDataGaussianWhite(num_unit3, pos_target3, ...
                                   echo_power_dB3, noise_power_dB3);
signal3_dB = 10 .* log10(signal3);

% data4:
%   # of units: 200; noise power1: 20 dB; noise power2: 30 dB
%   target position: [50]; echo wave power: [35]dB
%   probability of false alarm: 1e-3
num_unit4_1 = 100;
num_unit4_2 = 100;
num_unit4 = num_unit4_1 + num_unit4_2;
pos_target4_1 = 50;
pos_target4_2 = [];
echo_power_dB4_1 = 35;
echo_power_dB4_2 = [];
noise_power_dB4_1 = 20;
noise_power_dB4_2 = 30;
Pfa4 = 1e-3;
signal4_1 = generateDataGaussianWhite(num_unit4_1, pos_target4_1, ...
                                   echo_power_dB4_1, noise_power_dB4_1);
signal4_2 = generateDataGaussianWhite(num_unit4_2, pos_target4_2, ...
                                   echo_power_dB4_2, noise_power_dB4_2);
signal4 = [signal4_1; signal4_2];
signal4_dB = 10 .* log10(signal4);

% data5:
%   # of units: 200; noise power1: 20 dB; noise power2: 30 dB
%   target position: [95]; echo wave power: [35]dB
%   probability of false alarm: 1e-3
num_unit5_1 = 100;
num_unit5_2 = 100;
num_unit5 = num_unit5_1 + num_unit5_2;
pos_target5_1 = 95;
pos_target5_2 = [];
echo_power_dB5_1 = 35;
echo_power_dB5_2 = [];
noise_power_dB5_1 = 20;
noise_power_dB5_2 = 30;
Pfa5 = 1e-3;
signal5_1 = generateDataGaussianWhite(num_unit5_1, pos_target5_1, ...
                                   echo_power_dB5_1, noise_power_dB5_1);
signal5_2 = generateDataGaussianWhite(num_unit5_2, pos_target5_2, ...
                                   echo_power_dB5_2, noise_power_dB5_2);
signal5 = [signal5_1; signal5_2];
signal5_dB = 10 .* log10(signal5);

% data6:
%   # of units: 200; noise power1: 20 dB; noise power2: 30 dB
%   target position: [50, 58]; echo wave power: [35, 40]dB
%   probability of false alarm: 1e-3
num_unit6_1 = 100;
num_unit6_2 = 100;
num_unit6 = num_unit6_1 + num_unit6_2;
pos_target6_1 = [50; 58];
pos_target6_2 = [];
echo_power_dB6_1 = [35; 40];
echo_power_dB6_2 = [];
noise_power_dB6_1 = 20;
noise_power_dB6_2 = 30;
Pfa6 = 1e-3;
signal6_1 = generateDataGaussianWhite(num_unit6_1, pos_target6_1, ...
                                   echo_power_dB6_1, noise_power_dB6_1);
signal6_2 = generateDataGaussianWhite(num_unit6_2, pos_target6_2, ...
                                   echo_power_dB6_2, noise_power_dB6_2);
signal6 = [signal6_1; signal6_2];
signal6_dB = 10 .* log10(signal6);

% ======Processes of data1=================================================
% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data1\n");
[pos_detect, threshold] = cacfar(signal, Pfa, 10, 3);
threshold_dB = pow2db(threshold);

figure;
title('CFAR for data1');
hold on;
grid on;
plot(1:num_unit, signal_dB, 'k', 'LineWidth', 1);
plot(14 : num_unit-13, threshold_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect');
pause;

% ======Processes of data2=================================================
% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data2\n");
[pos_detect2, threshold2] = cacfar(signal2, Pfa2, 10, 3);
threshold2_dB = pow2db(threshold2);

figure;
title('CFAR for data2');
hold on;
grid on;
plot(1:num_unit2, signal2_dB, 'k', 'LineWidth', 1);
plot(14 : num_unit2-13, threshold2_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect2');
pause;

% ======Processes of data3=================================================
% ======CA-CFAR(10, 0)=====================================================
fprintf("===> CA-CFAR(10, 0) for data3\n");
[pos_detect3, threshold3] = cacfar(signal3, Pfa3, 10, 0);
threshold3_dB = pow2db(threshold3);

figure;
title('CFAR for data3');
hold on;
grid on;
plot(1:num_unit3, signal3_dB, 'k', 'LineWidth', 1);
plot(11 : num_unit3-10, threshold3_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect3');
% pause;

% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data3\n");
[pos_detect3_2, threshold3_2] = cacfar(signal3, Pfa3, 10, 3);
threshold3_dB_2 = pow2db(threshold3_2);

plot(14 : num_unit3-13, threshold3_dB_2, 'r', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect3_2');
pause;

% ======Processes of data4=================================================
% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data4\n");
[pos_detect4, threshold4] = cacfar(signal4, Pfa4, 10, 3);
threshold4_dB = pow2db(threshold4);

figure;
title('CFAR for data4');
hold on;
grid on;
plot(1:num_unit4, signal4_dB, 'k', 'LineWidth', 1);
plot(14 : num_unit4-13, threshold4_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect4');
pause;

% ======Processes of data5=================================================
% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data5\n");
[pos_detect5, threshold5] = cacfar(signal5, Pfa5, 10, 3);
threshold5_dB = pow2db(threshold5);

figure;
title('CFAR for data5');
hold on;
grid on;
plot(1:num_unit5, signal5_dB, 'k', 'LineWidth', 1);
plot(14 : num_unit5-13, threshold5_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect5');
pause;

% ======Processes of data6=================================================
% ======CA-CFAR(10, 3)=====================================================
fprintf("===> CA-CFAR(10, 3) for data6\n");
[pos_detect6, threshold6] = cacfar(signal6, Pfa6, 10, 3);
threshold6_dB = pow2db(threshold6);

figure;
title('CFAR for data6');
hold on;
grid on;
plot(1:num_unit6, signal6_dB, 'k', 'LineWidth', 1);
plot(14 : num_unit6-13, threshold6_dB, 'b', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect6');
% pause;

% ======SOCA-CFAR(10, 3)===================================================
fprintf("===> SOCA-CFAR(10, 3) for data6\n");
[pos_detect6_1, threshold6_1] = socacfar(signal6, Pfa6, 10, 3);
threshold6_dB_1 = pow2db(threshold6_1);

plot(14 : num_unit6-13, threshold6_dB_1, 'r', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect6_1');
% pause;

% ======GOCA-CFAR(10, 3)===================================================
fprintf("===> GOCA-CFAR(10, 3) for data6\n");
[pos_detect6_2, threshold6_2] = gocacfar(signal6, Pfa6, 10, 3);
threshold6_dB_2 = pow2db(threshold6_2);

plot(14 : num_unit6-13, threshold6_dB_2, 'g', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect6_2');
% pause;

% ======OS-CFAR(10, 3)===================================================
fprintf("===> OS-CFAR(10, 3) for data6\n");
[pos_detect6_3, threshold6_3] = oscfar(signal6, Pfa6, 10, 3, 10);
threshold6_dB_3 = pow2db(threshold6_3);

plot(14 : num_unit6-13, threshold6_dB_3, 'y', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect6_3');
% pause;

% ======S-CFAR(10, 3)===================================================
fprintf("===> S-CFAR(10, 3) for data6\n");
[pos_detect6_4, threshold6_4] = scfar(signal6, Pfa6, 10, 3, 30, 10);
threshold6_dB_4 = pow2db(threshold6_4);

plot(14 : num_unit6-13, threshold6_dB_4, 'm', 'LineWidth', 0.5);

fprintf("===> target position\n");
display(pos_detect6_4');
pause;