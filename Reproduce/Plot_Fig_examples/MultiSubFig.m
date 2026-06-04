clear; close all; clc;

width = 12;
height = 6.5;
fontsize = 14;
linewidth = 1;
markersize = 3;
legendfontsize  = 12;
xlabel_fontsize = 12;
ylabel_fontsize = 12;
title_fontsize = 10;
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');


fig = figure(1);
set(fig, 'Units', 'inches');
set(fig, 'Position', [1, 1, width, height]);
set(fig, 'Color', 'w');
set(fig, 'Renderer', 'painters');


rows = 2;
cols = 4;
t = tiledlayout(fig, rows, cols);
t.TileSpacing = 'tight';
t.Padding = 'tight';

x = 0:0.1:10;

for idx = 1:cols*rows
    ax = nexttile(t);
    y1 = exp(-0.15 * idx * x) .* abs(sin(x + 0.3 * idx));
    y2 = exp(-0.12 * idx * x) .* abs(cos(x + 0.2 * idx));
    semilogy(ax, x, y1 + 1e-4, '-o', 'LineWidth', linewidth, 'MarkerSize', markersize); hold(ax, 'on');
    semilogy(ax, x, y2 + 1e-4, '--s', 'LineWidth', linewidth, 'MarkerSize', markersize);
    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', fontsize, 'LineWidth', 1.2);
    set(ax,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

    axis(ax, [0 10 1e-4 1.2]);

    hx = xlabel(ax, 'SNR(dB)', 'FontSize', xlabel_fontsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    hy = ylabel(ax, 'WER', 'FontSize', ylabel_fontsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    % 控制 Xlabel/Ylabel 与坐标轴的距离
    % hx.Units = 'normalized';  hx.Position(2) = -0.11;
    % hy.Units = 'normalized';  hy.Position(1) = -0.11;
    title(ax, ['Subfigure ', num2str(idx)], 'FontSize', title_fontsize, 'Interpreter', 'latex');
    if idx == 1
        h_legend = legend(ax, 'Method 1', 'Method 2', 'FontSize',legendfontsize, 'FontWeight','normal', 'Location', 'southwest', 'Interpreter', 'latex');
        %set(h_legend,'FontName','宋体','FontSize',legendsize,'FontWeight','normal','LineWidth', 1, 'Location','SouthWest');
        %set(h_legend,'Interpreter','latex') %  'box','off');
    end
end

drawnow;

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, width, height]);
set(fig, 'PaperSize', [width, height]);
set(fig, 'PaperPositionMode', 'manual');

print(fig, 'Fig_2x3.pdf', '-dpdf', '-vector');
% print(fig, 'Fig_2x3.pdf', '-dpdf', '-painters');
print(fig, 'Fig_2x3.eps', '-depsc', '-vector');
print(fig, 'Fig_2x3.png', '-dpng', '-r300');