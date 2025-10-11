clear; close all; clc;
% 对CA CFAR, OS CFAR, SOCA CFAR, GOCA CFAR
% 杂波边缘的虚警概率曲线

MC_num = 100000; % Monte Carlo次数

num_unit = 200;
pos_target = [];
noise_power_dB1 = 20;
noise_power_dB2_list = 30;
Pfa = 1e-3;

[detection_times1, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, MC_num, num_unit, 1);
[detection_times2, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, MC_num, num_unit, 2);
[detection_times3, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, MC_num, num_unit, 3);
[detection_times4, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, MC_num, num_unit, 4);

figure;
subplot(211);
hold on;
grid on;
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times4(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
xlabel('距离单元');
ylabel('虚警概率P_{FA}');
legend('CA CFAR', 'OS CFAR', 'SOCA CFAR', 'GOCA CFAR');
xlim([0, num_unit]);

subplot(212);
hold on;
grid on;
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times4(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
xlabel('距离单元');
ylabel('虚警概率P_{FA}');
legend('CA CFAR', 'OS CFAR', 'SOCA CFAR', 'GOCA CFAR');
xlim([80, 120]);


function [detection_times, start_cell, stop_cell] = MCSim(noise_power_dB1, noise_power_dB2, Pfa, MC_num, num_unit, cfar_id)
    % cfar_id: 1-cacfar, 2-oscfar, 3-socacfar, 4-gocacfar
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

        if cfar_id == 1
            [pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
        elseif cfar_id == 2
            [pos, thres, start_cell, stop_cell] = oscfar(signal, Pfa, 10, 2, 15);
        elseif cfar_id == 3
            [pos, thres, start_cell, stop_cell] = socacfar(signal, Pfa, 10, 2);
        elseif cfar_id == 4
            [pos, thres, start_cell, stop_cell] = gocacfar(signal, Pfa, 10, 2);
        end
    
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