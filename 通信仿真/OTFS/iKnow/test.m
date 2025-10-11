
%% ML-CFAR检测仿真：低速小目标检测
clear; close all; clc;

%% 参数设置
N_range = 256;      % 距离单元数
N_doppler = 64;     % 多普勒单元数
P_fa_desired = 1e-4; % 期望虚警概率

% 目标参数
target_range = [80, 150, 200];      % 目标距离单元
target_doppler = [2, 3, -2];        % 目标多普勒单元（低速）
target_amplitude = [3.5, 3.0, 2.8]; % 目标幅度（小目标）

% CFAR参数
N_guard_range = 4;      % 距离维保护单元
N_guard_doppler = 8;    % 多普勒维保护单元
N_ref_range = 32;       % 距离维参考单元
N_ref_doppler = 16;      % 多普勒维参考单元

% 杂波参数
clutter_power = 1.0;    % 杂波功率
SNR_dB = 10;            % 信噪比(dB)

%% 生成距离-多普勒图
fprintf('生成距离-多普勒图...\n');
RD_map = generate_RD_map(N_range, N_doppler, clutter_power);

% 添加目标
for i = 1:length(target_range)
    range_idx = target_range(i);
    doppler_idx = target_doppler(i) + N_doppler/2 + 1; % 转换为正索引
    amplitude = target_amplitude(i);

    % 确保索引在范围内
    if range_idx > 0 && range_idx <= N_range && doppler_idx > 0 && doppler_idx <= N_doppler
        RD_map(range_idx, doppler_idx) = RD_map(range_idx, doppler_idx) + amplitude;
    end
end

% 添加噪声
noise_power = clutter_power / (10^(SNR_dB/10));
RD_map = RD_map + sqrt(noise_power/2) * (randn(size(RD_map)) + 1j * randn(size(RD_map)));

% 取幅度
RD_map_mag = abs(RD_map);

%% ML-CFAR检测器实现
fprintf('执行ML-CFAR检测...\n');
detection_map = ml_cfar_detector_2D(RD_map_mag, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired);

%% 传统CA-CFAR用于比较
fprintf('执行CA-CFAR检测用于比较...\n');
detection_map_ca = ca_cfar_detector_2D(RD_map_mag, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired);

%% 结果显示
figure('Position', [100, 100, 1200, 800]);

% 原始距离-多普勒图
subplot(2,3,1);
imagesc(1:N_doppler, 1:N_range, 20*log10(RD_map_mag));
colorbar; title('距离-多普勒图 (dB)'); xlabel('多普勒单元'); ylabel('距离单元');
hold on;
for i = 1:length(target_range)
    doppler_idx = target_doppler(i) + N_doppler/2 + 1;
    plot(doppler_idx, target_range(i), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;

% ML-CFAR检测结果
subplot(2,3,2);
imagesc(1:N_doppler, 1:N_range, detection_map);
colorbar; title('ML-CFAR检测结果'); xlabel('多普勒单元'); ylabel('距离单元');
hold on;
for i = 1:length(target_range)
    doppler_idx = target_doppler(i) + N_doppler/2 + 1;
    plot(doppler_idx, target_range(i), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;

% CA-CFAR检测结果
subplot(2,3,3);
imagesc(1:N_doppler, 1:N_range, detection_map_ca);
colorbar; title('CA-CFAR检测结果'); xlabel('多普勒单元'); ylabel('距离单元');
hold on;
for i = 1:length(target_range)
    doppler_idx = target_doppler(i) + N_doppler/2 + 1;
    plot(doppler_idx, target_range(i), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
hold off;

% 距离维切片比较
range_slice = target_range(1);
subplot(2,3,4);
plot(1:N_doppler, RD_map_mag(range_slice, :), 'b-', 'LineWidth', 2);
hold on;
ml_threshold = calculate_ml_threshold_line(RD_map_mag, range_slice, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired);
ca_threshold = calculate_ca_threshold_line(RD_map_mag, range_slice, N_guard_range, N_guard_doppler, N_ref_range, N_ref_doppler, P_fa_desired);
plot(1:N_doppler, ml_threshold, 'r--', 'LineWidth', 2, 'DisplayName', 'ML-CFAR阈值');
plot(1:N_doppler, ca_threshold, 'g--', 'LineWidth', 2, 'DisplayName', 'CA-CFAR阈值');
title(['距离单元 ', num2str(range_slice), ' 的检测情况']);
xlabel('多普勒单元'); ylabel('幅度');
legend('信号', 'ML-CFAR阈值', 'CA-CFAR阈值');
grid on;

% 性能统计
subplot(2,3,5);
[ml_stats, ca_stats] = calculate_detection_stats(detection_map, detection_map_ca, target_range, target_doppler, N_doppler);
performance_data = [ml_stats.detection_rate, ca_stats.detection_rate; 
                   ml_stats.false_alarms, ca_stats.false_alarms];
bar(performance_data');
set(gca, 'XTickLabel', {'检测率', '虚警数'});
ylabel('性能指标');
title('ML-CFAR vs CA-CFAR 性能比较');
legend('ML-CFAR', 'CA-CFAR');
grid on;

% 低速目标区域放大
subplot(2,3,6);
low_speed_doppler = (N_doppler/2-5):(N_doppler/2+5);
imagesc(low_speed_doppler, 1:N_range, RD_map_mag(:, low_speed_doppler));
colorbar; title('低速目标区域放大'); xlabel('多普勒单元'); ylabel('距离单元');
hold on;
for i = 1:length(target_range)
    doppler_idx = target_doppler(i) + N_doppler/2 + 1;
    if ismember(doppler_idx, low_speed_doppler)
        plot(doppler_idx, target_range(i), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    end
end
hold off;

%% 输出性能统计
fprintf('\n=== 检测性能统计 ===\n');
fprintf('ML-CFAR: 检测率 = %.2f%%, 虚警数 = %d\n', ml_stats.detection_rate*100, ml_stats.false_alarms);
fprintf('CA-CFAR: 检测率 = %.2f%%, 虚警数 = %d\n', ca_stats.detection_rate*100, ca_stats.false_alarms);

%% 函数定义
function RD_map = generate_RD_map(N_range, N_doppler, clutter_power)
    % 生成距离-多普勒图，包含瑞利分布杂波
    RD_map = sqrt(clutter_power/2) * (randn(N_range, N_doppler) + 1j * randn(N_range, N_doppler));
end

function detection_map = ml_cfar_detector_2D(data, guard_r, guard_d, ref_r, ref_d, P_fa)
    % 2D ML-CFAR检测器（基于瑞利分布假设）
    [N_range, N_doppler] = size(data);
    detection_map = zeros(size(data));

    % 计算阈值系数（瑞利分布）
    alpha = sqrt(-2 * log(P_fa));

    for i = 1 + guard_r + ref_r : N_range - guard_r - ref_r
        for j = 1 + guard_d + ref_d : N_doppler - guard_d - ref_d
            % 提取参考单元
            ref_cells = get_reference_cells(data, i, j, guard_r, guard_d, ref_r, ref_d);

            % ML估计：瑞利分布的尺度参数
            sigma_ml = sqrt(sum(ref_cells.^2) / (2 * length(ref_cells)));

            % 计算检测阈值
            threshold = alpha * sigma_ml;

            % 检测判决
            if data(i, j) > threshold
                detection_map(i, j) = 1;
            end
        end
    end
end

function detection_map = ca_cfar_detector_2D(data, guard_r, guard_d, ref_r, ref_d, P_fa)
    % 2D CA-CFAR检测器用于比较
    [N_range, N_doppler] = size(data);
    detection_map = zeros(size(data));

    % 计算阈值系数（基于瑞利分布假设）
    N_ref_total = (2*ref_r + 1)*(2*ref_d + 1) - (2*guard_r + 1)*(2*guard_d + 1);
    alpha = N_ref_total * (P_fa^(-1/N_ref_total) - 1);

    for i = 1 + guard_r + ref_r : N_range - guard_r - ref_r
        for j = 1 + guard_d + ref_d : N_doppler - guard_d - ref_d
            % 提取参考单元
            ref_cells = get_reference_cells(data, i, j, guard_r, guard_d, ref_r, ref_d);

            % CA-CFAR：平均功率估计
            Z = mean(ref_cells);

            % 计算检测阈值
            threshold = alpha * Z;

            % 检测判决
            if data(i, j) > threshold
                detection_map(i, j) = 1;
            end
        end
    end
end

function ref_cells = get_reference_cells(data, center_i, center_j, guard_r, guard_d, ref_r, ref_d)
    % 获取参考单元（排除保护单元）
    ref_cells = [];

    for i = center_i - ref_r - guard_r : center_i + ref_r + guard_r
        for j = center_j - ref_d - guard_d : center_j + ref_d + guard_d
            % 跳过保护单元
            if abs(i - center_i) <= guard_r && abs(j - center_j) <= guard_d
                continue;
            end
            % 跳过中心单元本身
            if i == center_i && j == center_j
                continue;
            end
            % 确保在边界内
            if i >= 1 && i <= size(data,1) && j >= 1 && j <= size(data,2)
                ref_cells = [ref_cells, data(i, j)];
            end
        end
    end
end

function threshold_line = calculate_ml_threshold_line(data, range_idx, guard_r, guard_d, ref_r, ref_d, P_fa)
    % 计算ML-CFAR在特定距离单元的阈值线
    [~, N_doppler] = size(data);
    threshold_line = zeros(1, N_doppler);
    alpha = sqrt(-2 * log(P_fa));

    for j = 1 + guard_d + ref_d : N_doppler - guard_d - ref_d
        ref_cells = get_reference_cells(data, range_idx, j, guard_r, guard_d, ref_r, ref_d);
        sigma_ml = sqrt(sum(ref_cells.^2) / (2 * length(ref_cells)));
        threshold_line(j) = alpha * sigma_ml;
    end
end

function threshold_line = calculate_ca_threshold_line(data, range_idx, guard_r, guard_d, ref_r, ref_d, P_fa)
    % 计算CA-CFAR在特定距离单元的阈值线
    [~, N_doppler] = size(data);
    threshold_line = zeros(1, N_doppler);
    N_ref_total = (2*ref_r + 1)*(2*ref_d + 1) - (2*guard_r + 1)*(2*guard_d + 1);
    alpha = N_ref_total * (P_fa^(-1/N_ref_total) - 1);

    for j = 1 + guard_d + ref_d : N_doppler - guard_d - ref_d
        ref_cells = get_reference_cells(data, range_idx, j, guard_r, guard_d, ref_r, ref_d);
        Z = mean(ref_cells);
        threshold_line(j) = alpha * Z;
    end
end

function [ml_stats, ca_stats] = calculate_detection_stats(ml_detection, ca_detection, target_range, target_doppler, N_doppler)
    % 计算检测性能统计

    % 真实目标位置
    true_targets = zeros(size(ml_detection));
    for i = 1:length(target_range)
        doppler_idx = target_doppler(i) + N_doppler/2 + 1;
        if doppler_idx > 0 && doppler_idx <= N_doppler
            true_targets(target_range(i), doppler_idx) = 1;
        end
    end

    % ML-CFAR统计
    ml_detected = sum(ml_detection(:) & true_targets(:));
    ml_total_targets = sum(true_targets(:));
    ml_stats.detection_rate = ml_detected / ml_total_targets;
    ml_stats.false_alarms = sum(ml_detection(:)) - ml_detected;

    % CA-CFAR统计
    ca_detected = sum(ca_detection(:) & true_targets(:));
    ca_total_targets = sum(true_targets(:));
    ca_stats.detection_rate = ca_detected / ca_total_targets;
    ca_stats.false_alarms = sum(ca_detection(:)) - ca_detected;
end