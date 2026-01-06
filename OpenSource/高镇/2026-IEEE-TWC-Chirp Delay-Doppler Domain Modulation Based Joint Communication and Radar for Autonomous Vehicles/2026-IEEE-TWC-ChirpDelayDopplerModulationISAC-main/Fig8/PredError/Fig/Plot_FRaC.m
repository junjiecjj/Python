%% Plot_FRaC_Advanced.m
% 读取多个JSON文件，对比不同预测误差和调制间距下的 BER / SER 曲线
%
% 需要 MATLAB R2016b+ (支持 jsondecode)
%
% 更新日志:
% - 支持从多个JSON文件读取数据
% - 自动为4种情况 x 3种模式 = 12条曲线分配样式
% - 绘图样式规则:
%   - 不同情况 -> 不同颜色
%   - 同一情况 -> 相同颜色, 不同线型/标记
% - 根据新的Python代码，修正了BER计算中的比特数分母

clear; clc; close all

%% ① 定义所有实验情况和绘图样式
% 在此集中管理所有情况，方便未来扩展
conditions = [
    struct('label', '[Pred. Err=0, \Delta=1]',   'file', 'FRaC_0d0_1.json', 'color', 'b');
    struct('label', '[Pred. Err=0.1, \Delta=1]', 'file', 'FRaC_0d1_1.json', 'color', 'r');
    struct('label', '[Pred. Err=0.2, \Delta=1]', 'file', 'FRaC_0d2_1.json', 'color', [0 .6 0]); % Dark Green    
];

% 定义三种模式及其线型和标记
modes      = {'spatial', 'qpsk', 'all'};
lineStyles = {'--', ':', '-'}; % 虚线, 点线, 实线
markers    = {'s', '^', 'o'}; % 方块, 三角, 圆圈

% 总迭代次数 (用于计算BER/SER)
IterAll = 10000;

%% ② 初始化图像
set(0,'defaultfigurecolor','w');
set(0,'defaultAxesFontName','Times New Roman', 'defaultAxesFontSize', 14);

% BER 图像
f_ber = figure('Name', 'BER Comparison');
hold on; grid on;
set(gca, 'YScale', 'log');
title('Bit Error Rate (BER) Comparison');
xlabel('SNR [dB]');
ylabel('BER');

% SER 图像
f_ser = figure('Name', 'SER Comparison');
hold on; grid on;
set(gca, 'YScale', 'log');
title('Symbol Error Rate (SER) Comparison');
xlabel('SNR [dB]');
ylabel('SER');

%% ③ 循环读取数据并绘图
snrs = -30:5:-5;
% 获取当前脚本所在的文件夹路径
[scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
% 构建上一级文件夹的路径
jsonParentDir = fullfile(scriptDir, '..');
% 外层循环: 遍历每一种情况 (a, b, c, d)
for i = 1:numel(conditions)
    cond = conditions(i);
    
 % 构建JSON文件的完整路径
    jsonFullPath = fullfile(jsonParentDir, cond.file);

    % 读取对应的JSON文件
    try
        res = jsondecode(fileread(jsonFullPath));
        fprintf('成功读取文件: %s\n', jsonFullPath);
    catch ME
        warning('无法读取文件: %s. 跳过此情况。\n错误信息: %s', jsonFullPath, ME.message);
        continue; % 如果文件不存在则跳过
    end
    
    % 内层循环: 遍历每一种模式 (spatial, qpsk, all)
    for j = 1:numel(modes)
        m = modes{j};

        % --- 新增代码: 在循环开始时就定义好 mode_label ---
        switch m
            case 'qpsk',    mode_label = 'QPSK-only';
            case 'spatial', mode_label = 'Spatial-only';
            case 'all',     mode_label = 'All-bits';
        end
        % ---------------------------------------------------

        ber = nan(size(snrs));
        ser = nan(size(snrs));

        % 从JSON中提取数据
        for k = 1:numel(snrs)
            key = matlab.lang.makeValidName(num2str(snrs(k)));

            if isfield(res, m) && isfield(res.(m), key)
                data = res.(m).(key);
                switch m
                    case 'qpsk'
                        ber(k) = data.BER_QPSK   / (IterAll * 4);
                        ser(k) = data.SER_QPSK   / IterAll;
                    case 'spatial'
                        ber(k) = data.BER_AntSel / (IterAll * 2);
                        ser(k) = data.SER_AntSel / IterAll;
                    case 'all'
                        ber(k) = data.BER        / (IterAll * 7);
                        ser(k) = data.SER        / IterAll;
                end
            end
        end
        
        % --- 绘图 ---
        plot_style = struct(...
            'Color', cond.color, ...
            'LineStyle', lineStyles{j}, ...
            'Marker', markers{j}, ...
            'MarkerSize', 8, ...
            'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s %s', cond.label, mode_label) ...
        );
        
        % 绘制BER
        figure(f_ber); % 切换到BER图像
        plot(snrs, replaceZerosWithNaN(ber), ...
            'Color', plot_style.Color, 'LineStyle', plot_style.LineStyle, 'Marker', plot_style.Marker, ...
            'MarkerSize', plot_style.MarkerSize, 'LineWidth', plot_style.LineWidth, 'DisplayName', plot_style.DisplayName);
        
        % 绘制SER
        figure(f_ser); % 切换到SER图像
        plot(snrs, replaceZerosWithNaN(ser), ...
            'Color', plot_style.Color, 'LineStyle', plot_style.LineStyle, 'Marker', plot_style.Marker, ...
            'MarkerSize', plot_style.MarkerSize, 'LineWidth', plot_style.LineWidth, 'DisplayName', plot_style.DisplayName);
    end
end

% ④ 完善图像
figure(f_ber);
legend('show', 'Location', 'southwest', 'FontSize', 11);
ylim([1e-5, 1]);

figure(f_ser);
legend('show', 'Location', 'southwest', 'FontSize', 11);
ylim([1e-5, 1]);

fprintf('绘图完成。\n');

%% 辅助函数
function v = replaceZerosWithNaN(v)
% 把 0 置 NaN，便于对数坐标不报错
v(v==0) = NaN;
end