clear; close all; clc;
% CA CFAR示例 + 对比CA CFAR与阈值检测性能曲线

MC_num = 1000; % Monte Carlo次数

num_unit = 200;
pos_target = 100;
SNR_dB = linspace(0, 20, 100);
noise_power_dB = 20;
Pfa = 1e-5;

% 一个示例
pos = [];
tmpi = 1;
echo_power_dB = noise_power_dB + 15;
while length(pos) <= 1
    if mod(tmpi, 1000) == 0
        fprintf("times: %d\n", tmpi);
    end
    signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                    echo_power_dB, noise_power_dB);
    [pos, thres, start_cell, stop_cell] = cacfar(signal, Pfa, 10, 2);
end
figure;
hold on;
grid on;
plot(1:num_unit, pow2db(signal), 'k-', 'linewidth', 0.5);
plot(start_cell:stop_cell, pow2db(thres), 'k--', 'linewidth', 1);
plot(pos, pow2db(signal(1, pos)), 'ro', 'markersize', 10);
legend('信号', 'CA CFAR阈值', '检测到目标');
xlabel('距离单元');
ylabel('幅度(dB)');
pause;

% Monte Carlo计算性能曲线
detection_num = zeros(size(SNR_dB));
PD = [];

for ii = 1:length(SNR_dB)
    fprintf("SNR(dB) = %f\n", SNR_dB(ii));
    for mc = 1:MC_num
        echo_power_dB = noise_power_dB + SNR_dB(ii);
        signal = generateDataGaussianWhite(num_unit, pos_target, ...
                                        echo_power_dB, noise_power_dB);
        pos = cacfar(signal, Pfa, 10, 2);
        if length(find(pos == pos_target))
            detection_num(ii) = detection_num(ii) + 1;
        end
    end
end

PD = detection_num ./ MC_num;
figure;
plot(SNR_dB, PD, 'k-', 'linewidth', 1);
hold on;
grid on;
addpath('../threshold_detection');
plot(SNR_dB, swerling0(SNR_dB, Pfa, 1), 'k--', 'linewidth', 1);
xlabel('SNR(dB)');
ylabel('检测概率P_D');
legend('CA CFAR检测', '阈值检测');
% plot(pow2db(signal));