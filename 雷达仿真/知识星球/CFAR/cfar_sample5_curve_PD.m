clear; close all; clc;
% 对CA CFAR, OS CFAR, SOCA CFAR, GOCA CFAR
% 杂波边缘处目标的检测概率曲线

MC_num = 1000; % Monte Carlo次数

num_unit = 200;
noise_power_dB1 = 20;
Pfa = 1e-3;

[detection_times1, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, 15, MC_num, num_unit, 1);
[detection_times2, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, 15, MC_num, num_unit, 2);
[detection_times3, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, 15, MC_num, num_unit, 3);
[detection_times4, start_cell, stop_cell] = MCSim(noise_power_dB1, 30, Pfa, 15, MC_num, num_unit, 4);

figure;
subplot(211);
grid on;
hold on;
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times4(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
xlim([0, num_unit]);
xlabel('距离单元');
ylabel('检测概率P_D');
legend('CA CFAR', 'OS CFAR', 'SOCA CFAR', 'GOCA CFAR');

subplot(212);
grid on;
hold on;
plot(start_cell:stop_cell, detection_times1(1, start_cell:stop_cell) ./ MC_num, 'k-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times2(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times3(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
plot(start_cell:stop_cell, detection_times4(1, start_cell:stop_cell) ./ MC_num, '-', 'linewidth', 1);
xlim([80, 120]);
xlabel('距离单元');
ylabel('检测概率P_D');
legend('CA CFAR', 'OS CFAR', 'SOCA CFAR', 'GOCA CFAR');



function [detection_times, start_cell, stop_cell] = MCSim(noise_power_dB1, noise_power_dB2, Pfa, ...
                                                          pos_SNR_dB, MC_num, num_unit, cfar_id)
    % cfar_id: 1-cacfar, 2-oscfar, 3-socacfar, 4-gocacfar
    detection_times = zeros(1, num_unit);

    for pos_target = 12:200-11
        fprintf("target position: %d\n", pos_target);
        for ii = 1:MC_num
            if pos_target <= 100
                signal1 = generateDataGaussianWhite(100, pos_target, ...
                                                    pos_SNR_dB + noise_power_dB1, noise_power_dB1);
                signal2 = generateDataGaussianWhite(100, [], ...
                                                    [], noise_power_dB2);
            else
                signal1 = generateDataGaussianWhite(100, [], ...
                                                    [], noise_power_dB1);
                signal2 = generateDataGaussianWhite(100, pos_target - 100, ...
                                                    pos_SNR_dB + noise_power_dB2, noise_power_dB2);
            end
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

            if ~isempty(find(pos == pos_target))
                detection_times(1, pos_target) = detection_times(1, pos_target) + 1;
            end
        end
    end
end