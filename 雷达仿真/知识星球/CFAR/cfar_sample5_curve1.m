clear; close all; clc;
% 对CA CFAR, OS CFAR, SOCA CFAR, GOCA CFAR
% 存在目标遮蔽时，不同SNR下的检测概率曲线

MC_num = 1000; % Monte Carlo次数

num_unit = 200;
pos_target = [95, 100];
SNR_dB = linspace(5, 25, 50);
noise_power_dB = 20;
Pfa = 1e-5;

figure;
hold on;
grid on;
[PD_95_1, PD_100_1] = fun1_cfar(num_unit, SNR_dB, pos_target, noise_power_dB, Pfa, MC_num, 1);
plot(SNR_dB, PD_95_1, 'k-', 'linewidth', 1);

[PD_95_2, PD_100_2] = fun1_cfar(num_unit, SNR_dB, pos_target, noise_power_dB, Pfa, MC_num, 2);
plot(SNR_dB, PD_95_2, '-', 'linewidth', 1);

[PD_95_3, PD_100_3] = fun1_cfar(num_unit, SNR_dB, pos_target, noise_power_dB, Pfa, MC_num, 3);
plot(SNR_dB, PD_95_3, '-', 'linewidth', 1);

[PD_95_4, PD_100_4] = fun1_cfar(num_unit, SNR_dB, pos_target, noise_power_dB, Pfa, MC_num, 4);
plot(SNR_dB, PD_95_4, '-', 'linewidth', 1);

xlabel('SNR(dB)');
ylabel('检测概率P_D');
legend('CA CFAR', 'OS CFAR', 'SOCA CFAR', 'GOCA CFAR');


function [PD_95, PD_100] = fun1_cfar(num_unit, SNR_dB, pos_target, noise_power_dB, Pfa, MC_num, cfar_id)
    % 控制100处的目标SNR为15dB，95处的目标SNR在0->20dB之间
    % cfar_id: 1-cacfar, 2-oscfar, 3-socacfar, 4-gocacfar
    detection_times_95  = zeros(1, length(SNR_dB));
    detection_times_100 = zeros(1, length(SNR_dB));

    % figure;
    for ii = 1:length(SNR_dB)
        fprintf("SNR = %f dB\n", SNR_dB(ii));
        echo_power_dB = [noise_power_dB + SNR_dB(ii), noise_power_dB + 15];
        for mc = 1:MC_num
            signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                               echo_power_dB, noise_power_dB);

            if cfar_id == 1
                [pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
            elseif cfar_id == 2
                [pos, thres, start_cell, stop_cell] = oscfar(signal, Pfa, 10, 2, 15);
            elseif cfar_id == 3
                [pos, thres, start_cell, stop_cell] = socacfar(signal, Pfa, 10, 2);
            elseif cfar_id == 4
                [pos, thres, start_cell, stop_cell] = gocacfar(signal, Pfa, 10, 2);
            end

            if ~isempty(find(pos == 95))
                detection_times_95(ii) = detection_times_95(ii) + 1;
            end
            if ~isempty(find(pos == 100))
                detection_times_100(ii) = detection_times_100(ii) + 1;
            end
        end
    end
    
    PD_95 = detection_times_95 ./ MC_num;
    PD_100 = detection_times_100 ./ MC_num;
end