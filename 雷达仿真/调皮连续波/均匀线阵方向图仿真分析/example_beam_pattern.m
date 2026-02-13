%% example_beam_pattern.m
% EXAMPLE_BEAM_PATTERN 天线阵列波束方向图示例
%
% 功能描述:
%   该示例演示如何使用MATLAB版本的4D雷达仿真代码
%   分析ULA阵列的波束方向图特性。
% 运行环境: MATLAB R2020a或更高版本

clear; close all; clc;

%% 添加路径
addpath('./functions');

fprintf('========================================\n');
fprintf('  天线阵列波束方向图分析示例\n');
fprintf('========================================\n\n');

%% 1. 创建ULA阵列
fprintf('步骤1: 创建ULA阵列...\n');
nr = 7;      % 阵元数量
d = 0.5;     % 归一化间距(半波长)
theta_0 = 0; % 目标角度
ula = create_ula(nr, d);

fprintf('ULA配置: %d阵元, 间距=%.2fλ\n\n', nr, d);

%% 2. 计算波束方向图(波束指向0度)
fprintf('步骤2: 计算波束方向图(波束指向0度)...\n');
% 解析解
[p_analytical, angle_bins] = get_beam_pattern(ula, theta_0);
% FFT测角
coef = get_steering_vector(ula, theta_0);
[p_fft, angle_bins_fft] = get_beam_pattern_fft(ula, coef);

angle_bins_deg = rad2deg(angle_bins);
angle_bins_fft_deg = rad2deg(angle_bins_fft);

%% 3. 绘制波束方向图对比
fprintf('步骤3: 绘制波束方向图(解析解 vs FFT)...\n');
figure('Name', 'Beam Pattern Comparison', 'Position', [100, 100, 800, 600]);
hold on;
plot(angle_bins_deg, p_analytical, 'b-', 'LineWidth', 1.5, 'DisplayName', '解析解');
plot(angle_bins_fft_deg, p_fft, 'r--', 'LineWidth', 1.5, 'DisplayName', 'FFT方法');
grid on;
xlabel('角度 [deg]');
ylabel('波束方向图 [dB]');
title(sprintf('ULA波束方向图 (N=%d, d=%.1fλ, θ_0=%.0f°)', nr, d, rad2deg(theta_0)));
legend('Location', 'Best');
xlim([-90, 90]);
ylim([-40, 0]);

%% 4. 极坐标显示:对比
fprintf('步骤4: 极坐标显示波束方向图...\n');
figure('Name', 'Beam Pattern - Polar', 'Position', [150, 100, 600, 600]);
polarplot(deg2rad(angle_bins_deg), p_analytical, 'b-', 'LineWidth', 1.5);
hold on
polarplot(deg2rad(angle_bins_fft_deg), p_fft, 'r--', 'LineWidth', 1.5);
thetalim([-90, 90]);
rlim([-40, 0]);
title('波束方向图(极坐标)');
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
legend('解析解','FFT');

%% 5. 分析不同间距的影响
fprintf('\n步骤5: 分析不同间距的影响...\n');
d_values = [0.1, 0.3, 0.5, 0.8];
figure('Name', 'Effect of Element Spacing', 'Position', [200, 100, 800, 600]);
hold on;
colors = lines(length(d_values));

for i = 1:length(d_values)
    ula_temp = create_ula(nr, d_values(i));
    [p_temp, angles_temp] = get_beam_pattern(ula_temp, 0);
    plot(rad2deg(angles_temp), p_temp, 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('d = %.1fλ', d_values(i)));
end
grid on;
xlabel('角度 [deg]');
ylabel('波束方向图 [dB]');
title('不同阵元间距的波束方向图');
legend('Location', 'Best');
xlim([-90, 90]);
ylim([-40, 0]);

%% 6. 分析栅瓣现象
fprintf('\n步骤6: 分析栅瓣现象(d=0.8λ)...\n');
theta_0_deg = 35;
theta_0 = deg2rad(theta_0_deg);

ula_normal = create_ula(nr, 0.5);
ula_grating = create_ula(nr, 0.8);

[p_normal, angles_normal] = get_beam_pattern(ula_normal, theta_0);
[p_grating, angles_grating] = get_beam_pattern(ula_grating, theta_0);

figure('Name', 'Grating Lobe Analysis', 'Position', [250, 100, 800, 600]);
hold on;
plot(rad2deg(angles_normal), p_normal, 'b-', 'LineWidth', 1.5, 'DisplayName', 'd=0.5λ');
plot(rad2deg(angles_grating), p_grating, 'r--', 'LineWidth', 1.5, 'DisplayName', 'd=0.8λ');
xline(theta_0_deg, 'k--', 'LineWidth', 1, 'DisplayName', sprintf('指向角 θ_0=%.0f°', theta_0_deg));
grid on;
xlabel('角度 [deg]');
ylabel('波束方向图 [dB]');
title('栅瓣现象分析');
legend('Location', 'Best');
xlim([-90, 90]);
ylim([-40, 0]);

fprintf('\n注意: 当d=0.8λ时，在约40°处出现栅瓣\n');

%% 7. 间距与角度热图分析
fprintf('\n步骤7: 间距与角度热图分析...\n');
% 参数设置
theta_deg = linspace(-80, 80, 200);
theta_rad = deg2rad(theta_deg);
d_lambda = linspace(0.01, 2, 100);
N = 7;  % 7元阵列

% 计算波束响应
response = zeros(length(d_lambda), length(theta_deg));
for i = 1:length(d_lambda)
    d_val = d_lambda(i);
    for j = 1:length(theta_rad)
        theta = theta_rad(j);
        % 计算波束方向图
        if abs(sin(theta)) < 1e-10
            response(i, j) = 1;
        else
            numerator = sin(N * pi * d_val * sin(theta));
            denominator = N * sin(pi * d_val * sin(theta));
            if abs(denominator) < 1e-10
                response(i, j) = 1;
            else
                response(i, j) = abs(numerator / denominator)^2;
            end
        end
    end
end
response_dB = 10 * log10(response);

% 绘制热图
figure('Name', 'Spacing vs Angle Heatmap', 'Position', [300, 100, 900, 600]);
imagesc(theta_deg, d_lambda, response_dB);
colormap('turbo');
colorbar;
caxis([-40, 0]);
axis xy;
xlabel('角度 θ [deg]');
ylabel('阵列间距 d/λ');
title('不同阵列间距的波束方向图热图');

%% 8. 窗函数对波束方向图的影响
fprintf('\n步骤8: 分析窗函数对波束方向图的影响...\n');
nr_win = 32;
ula_win = create_ula(nr_win, 0.5);
theta_0 = 0;
coef = get_steering_vector(ula_win, theta_0);

% 不同窗函数
win_rect = ones(nr_win, 1);
win_hann = hann(nr_win);
win_hamming = hamming(nr_win);
win_blackman = blackman(nr_win);

[p_rect, angles_win] = get_beam_pattern_fft(ula_win, coef .* win_rect);
[p_hann, ~] = get_beam_pattern_fft(ula_win, coef .* win_hann);
[p_hamming, ~] = get_beam_pattern_fft(ula_win, coef .* win_hamming);
[p_blackman, ~] = get_beam_pattern_fft(ula_win, coef .* win_blackman);

% 绘制窗函数形状
figure('Name', 'Window Function Shape', 'Position', [350, 100, 800, 600]);
t = 0:nr_win-1;
hold on;
plot(t, win_rect, 'b-', 'LineWidth', 1.5, 'DisplayName', '矩形窗');
plot(t, win_hann, 'r-', 'LineWidth', 1.5, 'DisplayName', '汉宁窗');
plot(t, win_hamming, 'g-', 'LineWidth', 1.5, 'DisplayName', '汉明窗');
plot(t, win_blackman, 'm-', 'LineWidth', 1.5, 'DisplayName', '布莱克曼窗');
grid on;
xlabel('阵元编号');
ylabel('系数');
title('窗函数形状对比');
legend('Location', 'Best');
xlim([0, nr_win-1]);

% 绘制窗函数对波束方向图的影响
figure('Name', 'Window Function Effect', 'Position', [400, 100, 800, 600]);
hold on;
plot(rad2deg(angles_win), p_rect, 'b-', 'LineWidth', 1.5, 'DisplayName', '矩形窗');
plot(rad2deg(angles_win), p_hann, 'r-', 'LineWidth', 1.5, 'DisplayName', '汉宁窗');
plot(rad2deg(angles_win), p_hamming, 'g-', 'LineWidth', 1.5, 'DisplayName', '汉明窗');
plot(rad2deg(angles_win), p_blackman, 'm-', 'LineWidth', 1.5, 'DisplayName', '布莱克曼窗');
grid on;
xlabel('角度 [deg]');
ylabel('波束方向图 [dB]');
title('窗函数对波束方向图的影响');
legend('Location', 'Best');
xlim([-90, 90]);
ylim([-60, 0]);

fprintf('\n窗函数效果:\n');
fprintf('  - 矩形窗: 主瓣最窄，旁瓣最高\n');
fprintf('  - 汉宁窗: 主瓣变宽，旁瓣降低\n');
fprintf('  - 汉明窗: 第一旁瓣最低\n');
fprintf('  - 布莱克曼窗: 旁瓣最低，主瓣最宽\n');

%% 9. 非均匀阵列分析 (ULA-7 vs MRA-4 vs ULA-4)
fprintf('\n步骤9: 非均匀阵列分析...\n');
theta_0 = 0;
nr_7 = 7;
nr_4 = 4;
d = 0.5;

ula_7 = create_ula(nr_7, d);
ula_4 = create_ula(nr_4, d);

% 7-ULA
coef_7 = get_steering_vector(ula_7, theta_0);
[p_fft_7, angle_bins_fft_7] = get_beam_pattern_fft(ula_7, coef_7);

% MRA-4 (最小冗余阵列)
mra_weight = [1, 1, 0, 0, 1, 0, 1]';
coef_mra = coef_7 .* mra_weight;
[p_fft_mra, ~] = get_beam_pattern_fft(ula_7, coef_mra);

% 4-ULA
coef_4 = get_steering_vector(ula_4, theta_0);
[p_fft_4, angle_bins_fft_4] = get_beam_pattern_fft(ula_4, coef_4);

% 绘制对比图
figure('Name', 'ULA vs MRA Comparison', 'Position', [450, 100, 800, 600]);
hold on;
plot(rad2deg(angle_bins_fft_7), p_fft_7, 'b-', 'LineWidth', 1.5, 'DisplayName', 'ULA-7');
plot(rad2deg(angle_bins_fft_7), p_fft_mra, 'r-', 'LineWidth', 1.5, 'DisplayName', 'MRA-4');
plot(rad2deg(angle_bins_fft_4), p_fft_4, 'g-', 'LineWidth', 1.5, 'DisplayName', 'ULA-4');
grid on;
xlabel('角度 [deg]');
ylabel('波束方向图 [dB]');
title('ULA-7 vs MRA-4 vs ULA-4');
legend('Location', 'Best');
xlim([-90, 90]);
ylim([-40, 0]);

% 极坐标对比
figure('Name', 'ULA vs MRA Comparison - Polar', 'Position', [500, 100, 600, 600]);
polarplot(angle_bins_fft_7, p_fft_7, 'b-', 'LineWidth', 1.5, 'DisplayName', 'ULA-7');
hold on;
polarplot(angle_bins_fft_7, p_fft_mra, 'r-', 'LineWidth', 1.5, 'DisplayName', 'MRA-4');
polarplot(angle_bins_fft_4, p_fft_4, 'g-', 'LineWidth', 1.5, 'DisplayName', 'ULA-4');
thetalim([-90, 90]);
rlim([-40, 0]);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
title('ULA-7 vs MRA-4 vs ULA-4 (极坐标)');
legend('Location', 'Best');

%% 10. MRA-4多信号分析
fprintf('\n步骤10: MRA-4多信号分析...\n');
theta_0_deg_18 = 18;
theta_0_18 = deg2rad(theta_0_deg_18);

% MRA-4 18度
coef_18 = get_steering_vector(ula_7, theta_0_18);
coef_mra_18 = coef_18 .* mra_weight;
[p_fft_mra_18, ~] = get_beam_pattern_fft(ula_7, coef_mra_18);

% 绘制多信号分析
figure('Name', 'MRA-4 Multi-Signal Analysis', 'Position', [550, 100, 800, 600]);
hold on;
plot(rad2deg(angle_bins_fft_7), p_fft_mra, 'b--', 'LineWidth', 1.5, 'DisplayName', 'MRA-4 (0°)');
plot(rad2deg(angle_bins_fft_7), p_fft_mra_18, 'r--', 'LineWidth', 1.5, 'DisplayName', 'MRA-4 (18°)');
plot(rad2deg(angle_bins_fft_7), p_fft_mra + p_fft_mra_18, 'k-', 'LineWidth', 2, 'DisplayName', 'MRA-4 (0°+18°)');
grid on;
xlabel('角度 [deg]');
ylabel('响应 [dB]');
title('MRA-4多信号分析');
legend('Location', 'Best');
xlim([-20, 40]);
ylim([-40, 0]);

% 极坐标多信号分析
figure('Name', 'MRA-4 Multi-Signal Analysis - Polar', 'Position', [600, 100, 600, 600]);
polarplot(angle_bins_fft_7, p_fft_mra, 'b--', 'LineWidth', 1.5, 'DisplayName', 'MRA-4 (0°)');
hold on;
polarplot(angle_bins_fft_7, p_fft_mra_18, 'r--', 'LineWidth', 1.5, 'DisplayName', 'MRA-4 (18°)');
polarplot(angle_bins_fft_7, p_fft_mra + p_fft_mra_18, 'k-', 'LineWidth', 2, 'DisplayName', 'MRA-4 (0°+18°)');
thetalim([-90, 90]);
rlim([-40, 0]);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
title('MRA-4多信号分析 (极坐标)');
legend('Location', 'Best');

fprintf('\n========================================\n');
fprintf('  示例运行完成!\n');
fprintf('========================================\n');
