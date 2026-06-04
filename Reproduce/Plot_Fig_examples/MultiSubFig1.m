

clear; close all; clc;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

width = 12;
height = 6.5;
axis_fontsize = 12;
label_fontsize = 14;
title_fontsize = 14;
legend_fontsize = 11;
linewidth = 1.8;
markersize = 7;

fig = figure(1);
set(fig, 'Units', 'inches');
set(fig, 'Position', [1, 1, width, height]);
set(fig, 'Color', 'w');
set(fig, 'Renderer', 'painters');

left_margin = 0.055;
right_margin = 0.025;
bottom_margin = 0.1;
top_margin = 0.055;
horizontal_gap = 0.045;
vertical_gap = 0.045;

num_col = 4;
num_row = 1;
ax_width = (1 - left_margin - right_margin - (num_col - 1) * horizontal_gap) / num_col;
ax_height = (1 - bottom_margin - top_margin - (num_row - 1) * vertical_gap) / num_row;

xlabel_ypos = -0.1;
ylabel_xpos = -0.1; 

x = 0:0.1:10;

for row = 1:num_row
    for col = 1:num_col
        idx = (row - 1) * num_col + col;
        left = left_margin + (col - 1) * (ax_width + horizontal_gap);
        bottom = 1 - top_margin - row * ax_height - (row - 1) * vertical_gap;
        ax = axes(fig, 'Position', [left, bottom, ax_width, ax_height]);
        y1 = exp(-0.15 * idx * x) .* abs(sin(x + 0.3 * idx));
        y2 = exp(-0.12 * idx * x) .* abs(cos(x + 0.2 * idx));
        semilogy(ax, x, y1 + 1e-4, '-o', 'LineWidth', linewidth, 'MarkerSize', markersize); hold(ax, 'on');
        semilogy(ax, x, y2 + 1e-4, '--s', 'LineWidth', linewidth, 'MarkerSize', markersize);
        grid(ax, 'on');
        box(ax, 'on');

        set(ax, 'FontName', 'Times New Roman', 'FontSize', axis_fontsize, 'LineWidth', 1.2);
        set(ax, 'LineWidth', 1, 'GridLineStyle', '--', 'Gridalpha',0.2, 'GridLineWidth', 0.5, 'Layer','bottom');

        axis(ax, [0 10 1e-4 1.2]);

        hx = xlabel(ax, 'SNR(dB)', 'FontName', 'Times New Roman', 'FontSize', label_fontsize, 'Interpreter', 'latex');
        hy = ylabel(ax, 'WER', 'FontName', 'Times New Roman', 'FontSize', label_fontsize, 'Interpreter', 'latex');
        % ht = title(ax, ['Subfigure ', num2str(idx)], 'FontName', 'Times New Roman', 'FontSize', title_fontsize, 'Interpreter', 'latex');

        % hx.Units = 'normalized'; hx.Position(2) = xlabel_ypos;
        % hy.Units = 'normalized'; hy.Position(1) = ylabel_xpos;

        if idx == 1
            hlgd = legend(ax, 'Method 1', 'Method 2', 'Location', 'southwest', 'Interpreter', 'latex');
            set(hlgd, 'FontName', 'Times New Roman');
            set(hlgd, 'FontSize', legend_fontsize);
        end
    end
end

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, width, height]);
set(fig, 'PaperSize', [width, height]);
set(fig, 'PaperPositionMode', 'manual');

print(fig, 'Fig_2x3_manual.pdf', '-dpdf', '-painters');
print(fig, 'Fig_2x3_manual.eps', '-depsc', '-vector');
print(fig, 'Fig_2x3_manual.png', '-dpng', '-r300');