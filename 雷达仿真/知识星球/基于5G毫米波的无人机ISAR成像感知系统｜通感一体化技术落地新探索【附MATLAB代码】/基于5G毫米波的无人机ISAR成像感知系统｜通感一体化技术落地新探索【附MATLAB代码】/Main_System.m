%% 无人机成像感知系统 - 主控脚本
% =============================================
% 项目：基于5G基站的无人机成像感知
% 用途：整合ISAR成像、PSO优化、Kalman跟踪
% =============================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('  无人机成像感知系统 v1.0\n');
fprintf('  天眸通感团队 - 算法组\n');
fprintf('========================================\n\n');

%% ===== 参数配置 =====
fprintf('[1/4] 加载配置...\n');

config = struct();
config.fc = 28e9;           % 载波频率 28GHz
config.B = 400e6;           % 带宽 400MHz
config.PRF = 1000;          % 脉冲重复频率
config.T_obs = 2;           % 观测时间 2秒
config.target_range = 1000; % 初始距离
config.target_velocity = 10;% 速度
config.rotation_rate = 0.5; % 旋转速度

% PSO参数
config.pso_particles = 20;
config.pso_iterations = 30;

% Kalman参数
config.kalman_Q = diag([0.1, 0.1, 0.5, 0.5]);
config.kalman_R = 25 * eye(2);

fprintf('    配置完成！\n\n');

%% ===== ISAR成像 =====
fprintf('[2/4] ISAR成像模块...\n');

try
    % 调用封装的函数（Week 1需要实现）
    [echo_data, range_compressed, isar_raw] = module_isar_imaging(config);
    fprintf('    ✓ ISAR成像完成\n');
    fprintf('    图像大小: %d x %d\n', size(isar_raw,1), size(isar_raw,2));
catch ME
    fprintf('    ✗ ISAR失败: %s\n', ME.message);
    fprintf('    提示: 请先实现 module_isar_imaging.m\n');
    return;
end

%% ===== PSO优化 =====
fprintf('\n[3/4] PSO优化模块...\n');

try
    % 调用PSO优化（Week 2需要实现）
    [isar_optimized, best_params, entropy_history] = ...
        module_pso_optimization(range_compressed, config);
    fprintf('    ✓ PSO优化完成\n');
    fprintf('    最优参数: [%.4f, %.4f]\n', best_params);
catch ME
    fprintf('    ✗ PSO失败: %s\n', ME.message);
    fprintf('    使用原始图像...\n');
    isar_optimized = isar_raw;
    entropy_history = [];
end

%% ===== Kalman跟踪 =====
fprintf('\n[4/4] Kalman跟踪模块...\n');

try
    % 生成模拟测量（实际应从ISAR图像提取）
    dt = 0.1;
    t = 0:dt:config.T_obs;
    N = length(t);
    true_x = config.target_range + config.target_velocity * t;
    true_y = 5 * sin(2*pi*0.5*t);
    measurements = [true_x; true_y] + 5*randn(2,N);
    
    % 调用Kalman滤波（Week 2需要实现）
    [est_traj, true_traj, rmse] = module_kalman_tracking(measurements, config);
    fprintf('    ✓ Kalman跟踪完成\n');
    fprintf('    RMSE: %.2f 米\n', rmse);
catch ME
    fprintf('    ✗ Kalman失败: %s\n', ME.message);
end

%% ===== 结果可视化 =====
fprintf('\n生成可视化结果...\n');

figure('Name', '系统运行结果', 'Position', [50, 50, 1400, 900]);
set(gcf, 'Color', 'w');

% 子图1：原始ISAR（归一化后显示，caxis[-40,0]才有意义）
subplot(2,3,1);
imagesc(20*log10(abs(isar_raw)/max(abs(isar_raw(:)))+eps));
colormap(jet); colorbar;
title('原始ISAR图像', 'FontSize', 12);
caxis([-40, 0]);

% 子图2：优化后ISAR（归一化后显示）
subplot(2,3,2);
imagesc(20*log10(abs(isar_optimized)/max(abs(isar_optimized(:)))+eps));
colorbar;
title('PSO优化后', 'FontSize', 12);
caxis([-40, 0]);

% 子图3：熵值收敛
subplot(2,3,3);
if ~isempty(entropy_history)
    plot(entropy_history, 'b-', 'LineWidth', 2);
    xlabel('迭代次数'); ylabel('图像熵');
    title('PSO收敛曲线', 'FontSize', 12);
    grid on;
else
    text(0.5, 0.5, 'PSO未运行', 'HorizontalAlignment', 'center');
end

% 子图4：轨迹跟踪
subplot(2,3,[4,5]);
if exist('est_traj', 'var')
    plot(true_traj(1,:), true_traj(2,:), 'g-', 'LineWidth', 2); hold on;
    plot(measurements(1,:), measurements(2,:), 'r.', 'MarkerSize', 6);
    plot(est_traj(1,:), est_traj(2,:), 'b-', 'LineWidth', 2);
    legend('真实轨迹', '测量', 'Kalman估计');
    xlabel('X (米)'); ylabel('Y (米)');
    title('轨迹跟踪', 'FontSize', 12);
    grid on; axis equal;
else
    text(0.5, 0.5, 'Kalman未运行', 'HorizontalAlignment', 'center');
end

% 子图5：性能对比
subplot(2,3,6);
if exist('rmse', 'var')
    rmse_raw = sqrt(mean(sum((true_traj(1:2,:)-measurements).^2,1)));
    bar([rmse_raw, rmse]);
    set(gca, 'XTickLabel', {'原始', 'Kalman'});
    ylabel('RMSE (米)');
    title(sprintf('精度提升: %.1f%%', (1-rmse/rmse_raw)*100));
    grid on;
end

% 保存（使用脚本所在目录，避免 pwd 不对导致报错）
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir), script_dir = pwd; end
saveas(gcf, fullfile(script_dir, 'system_results.png'));
saveas(gcf, fullfile(script_dir, 'system_results.pdf'));
fprintf('    结果已保存！\n');

%% ===== 性能报告 =====
fprintf('\n========================================\n');
fprintf('           性能报告\n');
fprintf('========================================\n');
fprintf('成像性能:\n');
fprintf('  距离分辨率: %.2f cm\n', 3e8/(2*config.B)*100);
fprintf('  方位分辨率: %.2f cm\n', ...
    3e8/(config.fc*config.rotation_rate*config.T_obs)*100);
if exist('rmse', 'var')
    fprintf('\n跟踪性能:\n');
    fprintf('  Kalman RMSE: %.2f 米\n', rmse);
end
fprintf('========================================\n');
fprintf('系统运行完成！\n');
fprintf('========================================\n');
