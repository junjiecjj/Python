clc; clear; close all;

x = linspace(0, 2*pi, 200);

figure('Color', 'w');

% 字体设置
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');

fontSizeAxis  = 10;
fontSizeLabel = 10;
fontSizeTitle = 12;

% 更紧凑的子图位置
% Position = [left bottom width height]
pos1 = [0.08 0.58 0.40 0.32];
pos2 = [0.55 0.58 0.40 0.32];
pos3 = [0.08 0.15 0.40 0.32];
pos4 = [0.55 0.15 0.40 0.32];

ax1 = axes('Position', pos1, 'Color', 'w', ...
           'FontName', 'Times New Roman', 'FontSize', fontSizeAxis);
plot(x, sin(x), 'LineWidth', 1.5);
title('$\sin(x)$', 'Interpreter', 'latex', 'FontSize', fontSizeTitle);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
grid on;

ax2 = axes('Position', pos2, 'Color', 'w', ...
           'FontName', 'Times New Roman', 'FontSize', fontSizeAxis);
plot(x, cos(x), 'LineWidth', 1.5);
title('$\cos(x)$', 'Interpreter', 'latex', 'FontSize', fontSizeTitle);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
grid on;

ax3 = axes('Position', pos3, 'Color', 'w', ...
           'FontName', 'Times New Roman', 'FontSize', fontSizeAxis);
plot(x, sin(2*x), 'LineWidth', 1.5);
title('$\sin(2x)$', 'Interpreter', 'latex', 'FontSize', fontSizeTitle);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
grid on;

ax4 = axes('Position', pos4, 'Color', 'w', ...
           'FontName', 'Times New Roman', 'FontSize', fontSizeAxis);
plot(x, cos(2*x), 'LineWidth', 1.5);
title('$\cos(2x)$', 'Interpreter', 'latex', 'FontSize', fontSizeTitle);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', fontSizeLabel);
grid on;


width = 8;
height = 6;
% 强制修改图形窗口的实际尺寸
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1, 1, width, height]);
% 可选：同时设置打印属性，确保 print 命令也能生效
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
% 导出（三种格式）
exportgraphics(gcf, 'multi_subplot.pdf', 'ContentType', 'vector');
exportgraphics(gcf, 'multi_subplot.eps', 'ContentType', 'vector');
exportgraphics(gcf, 'multi_subplot.png', 'Resolution', 300);