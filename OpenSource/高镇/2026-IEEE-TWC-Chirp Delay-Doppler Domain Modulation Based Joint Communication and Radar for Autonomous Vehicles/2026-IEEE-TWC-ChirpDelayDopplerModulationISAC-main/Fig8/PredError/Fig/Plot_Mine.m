%% Plot_Mine_12curves.m — 从四个结果文件画 12 条曲线（BER & SER）
% 情况与文件名：
% a) Err=0.0, Δf=1 → Mine_0d0_1.json
% b) Err=0.1, Δf=1 → Mine_0d1_1.json
% c) Err=0.2, Δf=1 → Mine_0d2_1.json
% d) Err=0.2, Δf=2 → Mine_0d2_2.json
%
% 颜色：同一“情况”用同色；线型/marker：All、Delay-only、QPSK-only 各不相同

clear; clc; close all;

% ---------- 配置 ----------
snrs = -30:5:-5;                    % 与仿真一致的 SNR 采样
set(0,'defaultfigurecolor','w');

% 上一级目录路径
baseDir = fileparts(mfilename('fullpath')); if isempty(baseDir), baseDir = pwd; end
jsonDir = fullfile(baseDir, '..');  % 定义 JSON 文件所在文件夹

% 四个文件（文件名与“误差/间距”的映射）
files  = {'Mine_0d0_1.json','Mine_0d1_1.json','Mine_0d2_1.json','Mine_0d2_2.json'};
labels = {'Pred. Error=0, \Delta=1', 'Pred. Error=0.1, \Delta=1', ...
          'Pred. Error=0.2, \Delta=1', 'Pred. Error=0.2, \Delta=2'};

% 每种“情况”一组颜色（同一种情况=同色；不同情况=不同色）
cols = [0.0000 0.4470 0.7410;   % 蓝
        0.8500 0.3250 0.0980;   % 橙
        0.4660 0.6740 0.1880;   % 绿
        0.4940 0.1840 0.5560];  % 紫

% 读取四个结果
resCell = cellfun(@(fn) jsondecode(fileread(fullfile(jsonDir, fn))), ...
                  files, 'UniformOutput', false);

% 3 种类别（同一“情况”用同色，不同线型和 marker）
styles(1).name = 'All';        styles(1).ls = '-';  styles(1).mk = 'o';
styles(2).name = 'Delay-only'; styles(2).ls = '--'; styles(2).mk = 'x';
styles(3).name = 'QPSK-only';  styles(3).ls = ':';  styles(3).mk = 's';

LW = 2; MS = 8;

% 一个小工具：根据类别名从结构体取对应序列（字段名见引用的原脚本）
%   BER: case1_both.BER / case2_tone_only.BER / case3_qpsk_only.BER
%   SER: case1_both.SER / case2_tone_only.SER_tone / case3_qpsk_only.SER_qpsk
get_series = @(res, kind, metric) ...
    (strcmp(kind,'All')       * (strcmp(metric,'BER') * res.case1_both.BER     + strcmp(metric,'SER') * res.case1_both.SER)) ...
  + (strcmp(kind,'Delay-only')* (strcmp(metric,'BER') * res.case2_tone_only.BER+ strcmp(metric,'SER') * res.case2_tone_only.SER_tone)) ...
  + (strcmp(kind,'QPSK-only') * (strcmp(metric,'BER') * res.case3_qpsk_only.BER+ strcmp(metric,'SER') * res.case3_qpsk_only.SER_qpsk));

%% ---------- BER（12 条） ----------
figure; hold on; grid on; legend on;
set(gca,'FontName','Times New Roman','FontSize',16, 'YScale','log');
for i = 1:numel(resCell)
    r = resCell{i};
    % All / Delay-only / QPSK-only：同一“情况”用同色，不同线型和 marker
    semilogy(snrs, r.case1_both.BER,       'o-', 'Color', cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] All (BER)',       labels{i}));
    semilogy(snrs, r.case2_tone_only.BER,  'x--','Color', cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] Delay-only (BER)',labels{i}));
    semilogy(snrs, r.case3_qpsk_only.BER,  's:','Color',  cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] QPSK-only (BER)', labels{i}));
end
xlabel('SNR [dB]'); ylabel('BER');
title('BER vs SNR (12 curves)');
legend('Location','southwest','NumColumns',2);

%% ---------- SER（12 条） ----------
figure; hold on; grid on; legend on;
set(gca,'FontName','Times New Roman','FontSize',16, 'YScale','log');
for i = 1:numel(resCell)
    r = resCell{i};
    semilogy(snrs, r.case1_both.SER,            'o-', 'Color', cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] All',        labels{i}));
    semilogy(snrs, r.case2_tone_only.SER_tone,  'x--','Color', cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] Delay-only', labels{i}));
    semilogy(snrs, r.case3_qpsk_only.SER_qpsk,  's:','Color', cols(i,:), 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('[%s] QPSK-only',  labels{i}));
end
xlabel('SNR [dB]'); ylabel('SER');
title('SER vs SNR (12 curves)');
legend('Location','southwest','NumColumns',2);
