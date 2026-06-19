%% 报告配图一键生成脚本
% 运行方式：直接在MATLAB中运行（无需 cd 到 matlab_code 目录）
% 输出：4张独立PNG图片，对应技术报告中的4个插图位置
%
% 输出文件（位于脚本同级的 figures/ 目录下）：
%   figures/fig_isar_result.png       - ISAR成像结果（三子图）
%   figures/fig_pso_convergence.png   - PSO收敛曲线
%   figures/fig_kalman_tracking.png   - 卡尔曼滤波跟踪
%   figures/fig_confusion_matrix.png  - 分类混淆矩阵

clear; clc; close all;

fprintf('==========================================\n');
fprintf('  技术报告配图生成脚本\n');
fprintf('==========================================\n\n');

%% ===== 路径设置（核心修复：确保无论 pwd 在哪都能正确保存） =====
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
fig_dir = fullfile(script_dir, 'figures');
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% 确保模块函数能被找到
addpath(script_dir);

%% ===== 公共参数 =====
config = struct();
config.fc = 28e9;
config.B = 400e6;
config.PRF = 1000;
config.T_obs = 2;
config.target_range = 1000;
config.target_velocity = 10;
config.rotation_rate = 0.5;
config.pso_particles = 20;
config.pso_iterations = 30;
config.kalman_Q = diag([0.1, 0.1, 0.5, 0.5]);
config.kalman_R = 25 * eye(2);

c = 3e8;
lambda = c / config.fc;

%% ===== 图1：ISAR成像结果 =====
fprintf('[1/4] 生成ISAR成像图...\n');

[echo_data, range_compressed, isar_image] = module_isar_imaging(config);

fig1 = figure('Name', 'ISAR成像结果', 'Position', [50, 50, 1500, 450], 'Color', 'w');

subplot(1,3,1);
imagesc(abs(echo_data));
colormap(jet); colorbar;
title('(a) 原始回波数据', 'FontSize', 13);
xlabel('慢时间（脉冲索引）', 'FontSize', 11);
ylabel('快时间（采样点）', 'FontSize', 11);

subplot(1,3,2);
rc_norm = 20*log10(abs(range_compressed)/max(abs(range_compressed(:))) + eps);
imagesc(rc_norm);
colorbar;
title('(b) 距离压缩后（dB）', 'FontSize', 13);
xlabel('慢时间（脉冲索引）', 'FontSize', 11);
ylabel('距离单元', 'FontSize', 11);
caxis([-40 0]);

subplot(1,3,3);
isar_norm = 20*log10(abs(isar_image)/max(abs(isar_image(:))) + eps);
imagesc(isar_norm);
colorbar;
title('(c) ISAR二维成像（dB）', 'FontSize', 13);
xlabel('多普勒频率', 'FontSize', 11);
ylabel('距离', 'FontSize', 11);
caxis([-40 0]);

out_path = fullfile(fig_dir, 'fig_isar_result.png');
saveas(fig1, out_path);
fprintf('  已保存 %s\n\n', out_path);

%% ===== 图2：PSO收敛曲线 =====
fprintf('[2/4] 生成PSO收敛曲线...\n');

[~, ~, entropy_history] = module_pso_optimization(range_compressed, config);

fig2 = figure('Name', 'PSO收敛曲线', 'Position', [50, 50, 700, 500], 'Color', 'w');

plot(1:length(entropy_history), entropy_history, 'b-o', ...
    'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
xlabel('迭代次数', 'FontSize', 13);
ylabel('全局最优图像熵', 'FontSize', 13);
title('PSO优化收敛曲线', 'FontSize', 14);
grid on;

% 标注起止点
hold on;
plot(1, entropy_history(1), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
plot(length(entropy_history), entropy_history(end), 'gp', ...
    'MarkerSize', 14, 'MarkerFaceColor', 'g', 'LineWidth', 2);

% 计算改善百分比
improvement = (entropy_history(1) - entropy_history(end)) / entropy_history(1) * 100;
legend({sprintf('熵值曲线'), ...
        sprintf('初始熵 = %.4f', entropy_history(1)), ...
        sprintf('最终熵 = %.4f（降低 %.1f%%）', entropy_history(end), improvement)}, ...
    'FontSize', 11, 'Location', 'northeast');

out_path = fullfile(fig_dir, 'fig_pso_convergence.png');
saveas(fig2, out_path);
fprintf('  已保存 %s\n\n', out_path);

%% ===== 图3：卡尔曼滤波跟踪 =====
fprintf('[3/4] 生成卡尔曼跟踪图...\n');

dt = 0.1;
t = 0:dt:config.T_obs;
N = length(t);
true_x = config.target_range + config.target_velocity * t;
true_y = 5 * sin(2*pi*0.5*t);
measurements = [true_x; true_y] + 5*randn(2,N);

[est_traj, true_traj, rmse] = module_kalman_tracking(measurements, config);
rmse_raw = sqrt(mean(sum((true_traj(1:2,:) - measurements).^2, 1)));

fig3 = figure('Name', '卡尔曼滤波跟踪', 'Position', [50, 50, 1200, 500], 'Color', 'w');

% 左图：轨迹对比
subplot(1,2,1);
plot(true_traj(1,:), true_traj(2,:), 'g-', 'LineWidth', 2.5); hold on;
plot(measurements(1,:), measurements(2,:), 'r.', 'MarkerSize', 8);
plot(est_traj(1,:), est_traj(2,:), 'b-', 'LineWidth', 2);
legend('真实轨迹', '含噪测量', '卡尔曼估计', 'FontSize', 11, 'Location', 'best');
xlabel('X 位置（m）', 'FontSize', 12);
ylabel('Y 位置（m）', 'FontSize', 12);
title('(a) 目标轨迹跟踪对比', 'FontSize', 13);
grid on; axis equal;

% 右图：RMSE柱状图
subplot(1,2,2);
b = bar([rmse_raw, rmse], 0.5);
b.FaceColor = 'flat';
b.CData(1,:) = [0.85 0.33 0.1];  % 红色
b.CData(2,:) = [0.0  0.45 0.74]; % 蓝色
set(gca, 'XTickLabel', {'原始测量', '卡尔曼滤波'}, 'FontSize', 12);
ylabel('RMSE（m）', 'FontSize', 12);
title(sprintf('(b) 定位精度对比（提升 %.1f%%）', (1-rmse/rmse_raw)*100), 'FontSize', 13);
grid on;

% 在柱状图上标注数值
text(1, rmse_raw + 0.2, sprintf('%.2f m', rmse_raw), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(2, rmse + 0.2, sprintf('%.2f m', rmse), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

out_path = fullfile(fig_dir, 'fig_kalman_tracking.png');
saveas(fig3, out_path);
fprintf('  已保存 %s\n\n', out_path);

%% ===== 图4：分类混淆矩阵 =====
fprintf('[4/4] 生成分类混淆矩阵...\n');

% 生成合成数据
num_samples = 500;
num_classes = 5;
image_size = 64;

images = zeros(image_size, image_size, num_samples);
labels = zeros(num_samples, 1);

for i = 1:num_samples
    label = randi(num_classes);
    labels(i) = label;
    img = zeros(image_size, image_size);

    switch label
        case 1  % 小型
            [X, Y] = meshgrid(linspace(-1,1,image_size), linspace(-1,1,image_size));
            img = exp(-(X.^2 + Y.^2) / 0.1);
        case 2  % 中型
            img(20:45, 20:45) = 1.0;
        case 3  % 大型
            img(10:55, 10:55) = 0.8;
            img(30:35, :) = 1.0;
        case 4  % 固定翼
            img(30:35, :) = 1.0;
            img(:, 30:35) = 0.5;
        case 5  % 旋翼
            for angle = linspace(0, 2*pi, 4)
                x = round(32 + 15*cos(angle));
                y = round(32 + 15*sin(angle));
                x = max(1, min(image_size, x));
                y = max(1, min(image_size, y));
                img(max(1,y-5):min(image_size,y+5), ...
                   max(1,x-5):min(image_size,x+5)) = 0.8;
            end
    end

    % 加较强噪声 + 随机缩放/偏移，使类间特征有一定重叠
    img = img * (0.6 + 0.8*rand()) + randn(image_size, image_size) * 0.25;
    img = max(0, min(1, img));
    images(:, :, i) = img;
end

% 提取统计特征
has_ipt = exist('graycomatrix', 'file') == 2;
if has_ipt
    nf = 9;
else
    nf = 7;
end
features = zeros(num_samples, nf);

for i = 1:num_samples
    img = images(:, :, i);
    features(i, 1) = mean(img(:));
    features(i, 2) = std(img(:));
    features(i, 3) = max(img(:));
    features(i, 4) = min(img(:));
    features(i, 5) = var(img(:));
    if has_ipt
        glcm = graycomatrix(uint8(img * 255));
        stats = graycoprops(glcm);
        features(i, 6) = stats.Contrast;
        features(i, 7) = stats.Correlation;
        features(i, 8) = stats.Energy;
        features(i, 9) = stats.Homogeneity;
    else
        features(i, 6) = sum(abs(diff(img(:))));
        features(i, 7) = sum(abs(diff(img, 1, 2)), 'all');
    end
end

% 训练与预测
split_ratio = 0.7;
num_train = round(split_ratio * num_samples);
train_features = features(1:num_train, :);
train_labels = labels(1:num_train);
test_features = features(num_train+1:end, :);
test_labels = labels(num_train+1:end);

try
    model = fitcecoc(train_features, train_labels);
    pred_labels = predict(model, test_features);
catch
    % KNN 回退
    k = 5;
    pred_labels = zeros(size(test_labels));
    for idx = 1:size(test_features, 1)
        dists = sum((train_features - test_features(idx,:)).^2, 2);
        [~, sorted_idx] = sort(dists);
        nearest_labels = train_labels(sorted_idx(1:k));
        pred_labels(idx) = mode(nearest_labels);
    end
end

accuracy = sum(pred_labels == test_labels) / length(test_labels) * 100;

% 混淆矩阵
cm = zeros(num_classes, num_classes);
for idx = 1:length(test_labels)
    cm(test_labels(idx), pred_labels(idx)) = cm(test_labels(idx), pred_labels(idx)) + 1;
end
cm_norm = cm ./ max(sum(cm, 2), 1);

% 画图
fig4 = figure('Name', '混淆矩阵', 'Position', [50, 50, 700, 600], 'Color', 'w');

imagesc(cm_norm, [0 1]);

% 蓝白渐变色图：0=白色，1=深蓝，中间过渡自然
blue_cmap = [ones(64,1), linspace(1,0.2,64)', linspace(1,0.2,64)'; ...
             linspace(1,0.1,64)', linspace(0.2,0.3,64)', linspace(0.2,0.8,64)'];
colormap(blue_cmap);
colorbar;

class_names = {'小型', '中型', '大型', '固定翼', '旋翼'};
[rows, cols] = size(cm);
for i = 1:rows
    for j = 1:cols
        if cm_norm(i,j) > 0.6
            txt_color = 'white';
        else
            txt_color = 'black';
        end
        text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), cm_norm(i,j)*100), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', txt_color, 'FontSize', 11, 'FontWeight', 'bold');
    end
end

set(gca, 'XTick', 1:cols, 'XTickLabel', class_names, 'FontSize', 11);
set(gca, 'YTick', 1:rows, 'YTickLabel', class_names, 'FontSize', 11);
xlabel('预测类别', 'FontSize', 13);
ylabel('真实类别', 'FontSize', 13);
title(sprintf('分类混淆矩阵（准确率: %.1f%%）', accuracy), 'FontSize', 14);

out_path = fullfile(fig_dir, 'fig_confusion_matrix.png');
saveas(fig4, out_path);
fprintf('  已保存 %s\n', out_path);
fprintf('  准确率: %.2f%%\n\n', accuracy);

%% ===== 汇总 =====
fprintf('==========================================\n');
fprintf('  全部配图生成完毕！\n');
fprintf('==========================================\n');
fprintf('输出目录: %s\n', fig_dir);
fprintf('  fig_isar_result.png\n');
fprintf('  fig_pso_convergence.png\n');
fprintf('  fig_kalman_tracking.png\n');
fprintf('  fig_confusion_matrix.png\n');
fprintf('\n性能指标：\n');
fprintf('  距离分辨率: %.2f cm\n', c/(2*config.B)*100);
fprintf('  方位分辨率: %.4f cm\n', lambda/(2*config.rotation_rate*config.T_obs)*100);
fprintf('  PSO熵改善:  %.1f%%\n', improvement);
fprintf('  跟踪RMSE:   %.2f m（原始 %.2f m，提升 %.1f%%）\n', ...
    rmse, rmse_raw, (1-rmse/rmse_raw)*100);
fprintf('  分类准确率: %.2f%%\n', accuracy);
fprintf('==========================================\n');
