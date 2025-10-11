clear; close all; clc;
% 杂波边缘处的虚警现象

MC_num = 100000; % Monte Carlo次数

num_unit = 200;
pos_target = [];
noise_power_dB1 = 20;
noise_power_dB2_list = 30;
Pfa = 1e-3;

figure;
hold on;
grid on;
[detection_times1, start_cell, stop_cell] = MCSim(noise_power_dB1, 24, Pfa, MC_num, num_unit);
subplot(211);
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 0.5);
hold on;
subplot(212);
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 0.5);
hold on;
pause(0.1);

[detection_times2, start_cell, stop_cell] = MCSim(noise_power_dB1, 27, Pfa, MC_num, num_unit);
subplot(211);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, 'r-', 'linewidth', 0.5);
hold on;
subplot(212);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, 'r-', 'linewidth', 0.5);
hold on;
pause(0.1);

[detection_times3, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, MC_num, num_unit);
subplot(211);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, 'b-', 'linewidth', 0.5);
hold on;
subplot(212);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, 'b-', 'linewidth', 0.5);
hold on;

xlim([0, num_unit]);
% ylim([0, 1]);
subplot(211);
grid on;
xlabel('距离单元');
ylabel('虚警概率P_{FA}');
legend('24dB', '27dB', '30dB');
subplot(212);
grid on;
xlabel('距离单元');
ylabel('虚警概率P_{FA}');
legend('24dB', '27dB', '30dB');
xlim([80, 120]);

function [detection_times, start_cell, stop_cell] = MCSim(noise_power_dB1, noise_power_dB2, Pfa, MC_num, num_unit)
    detection_times = zeros(1, num_unit);

    for ii = 1:MC_num
        if mod(ii, 1000) == 0
            fprintf("times: %d\n", ii);
        end
        signal1 = generateDataGaussianWhite(100, [], ...
                                            [], noise_power_dB1);
        signal2 = generateDataGaussianWhite(100, [], ...
                                            [], noise_power_dB2);
        signal = [signal1, signal2];
        [pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
    
        if ~isempty(pos)
            % figure;
            % hold on;
            % grid on;
            % plot(pow2db(signal), 'k-', 'linewidth', 0.5);
            % plot(start_cell:stop_cell, pow2db(thres), 'k--', 'linewidth', 1);
            % plot(pos, pow2db(signal(1, pos)), 'ro', 'markersize', 10);
            % legend('信号', 'CA CFAR阈值', '检测到目标');
            % xlabel('距离单元');
            % ylabel('幅度(dB)');
            % pause;
            detection_times(1, pos) = detection_times(1, pos) + 1;
        end
    end
end