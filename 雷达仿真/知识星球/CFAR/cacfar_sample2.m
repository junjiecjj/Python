clear; close all; clc;
% 仿真CA CFAR的目标遮蔽效应

MC_num = 1000; % Monte Carlo次数

num_unit = 200;
pos_target = [95, 100];
SNR_dB = linspace(10, 20, 50);
noise_power_dB = 20;
Pfa = 1e-5;

% 展示一个目标遮蔽的示例
echo_power_dB = [noise_power_dB + 12, noise_power_dB + 15];
signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                   echo_power_dB, noise_power_dB);
[pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
figure;
hold on;
grid on;
plot(1:num_unit, pow2db(signal), 'k-', 'linewidth', 0.5);
plot(start_cell:stop_cell, pow2db(thres), 'k--', 'linewidth', 1);
if ~isempty(pos)
    plot(pos, pow2db(signal(1, pos)), 'ro', 'markersize', 10);
end
legend('信号', 'CA CFAR阈值', '检测到目标');
xlabel('距离单元');
ylabel('幅度(dB)');

% 控制100处的目标SNR为15dB，95处的目标SNR在0->20dB之间
detection_times_95  = zeros(1, length(SNR_dB));
detection_times_100 = zeros(1, length(SNR_dB));
PD_95 = [];
PD_100 = [];

% figure;
for ii = 1:length(SNR_dB)
    fprintf("SNR = %f dB\n", SNR_dB(ii));
    echo_power_dB = [noise_power_dB + SNR_dB(ii), noise_power_dB + 15];
    for mc = 1:MC_num
        signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                           echo_power_dB, noise_power_dB);
        [pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
        if ~isempty(find(pos == 95))
            detection_times_95(ii) = detection_times_95(ii) + 1;
        end
        if ~isempty(find(pos == 100))
            detection_times_100(ii) = detection_times_100(ii) + 1;
        end
        % hold off;
        % plot(1:num_unit, pow2db(signal), 'k-', 'linewidth', 0.5);
        % hold on;
        % plot(start_cell:stop_cell, pow2db(thres), 'k--', 'linewidth', 1);
        % if ~isempty(pos)
        %     plot(pos, pow2db(signal(1, pos)), 'ro', 'markersize', 10);
        % end
        % pause(0.1);
    end
end

PD_95 = detection_times_95 ./ MC_num;
PD_100 = detection_times_100 ./ MC_num;
figure;
hold on;
grid on;
plot(SNR_dB, PD_95, 'k-', 'linewidth', 1);
plot(SNR_dB, PD_100, 'k--', 'linewidth', 1);
xlabel('SNR(dB)');
ylabel('检测概率P_D');
legend('95位置检测概率', '100位置检测概率');