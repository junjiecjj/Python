clear; close all; clc;
width = 12;
height = 6.5;
fontsize = 14;
linewidth = 1;
markersize = 3;
legendfontsize = 12;
label_fontsize = 16;
title_fontsize = 10;

set(groot, 'defaultAxesToolbarVisible', 'off');
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
t.TileSpacing = 'compact';
t.Padding = 'compact';

x = 0:0.1:10;

for idx = 1:rows * cols
    ax = nexttile(t);
    if isprop(ax, 'Toolbar')
        ax.Toolbar.Visible = 'off';
    end

    y1 = exp(-0.15 * idx * x) .* abs(sin(x + 0.3 * idx));
    y2 = exp(-0.12 * idx * x) .* abs(cos(x + 0.2 * idx));

    semilogy(ax, x, y1 + 1e-4, '-o', 'LineWidth', linewidth, 'MarkerSize', markersize);
    hold(ax, 'on');
    semilogy(ax, x, y2 + 1e-4, '--s', 'LineWidth', linewidth, 'MarkerSize', markersize);

    grid(ax, 'on');
    box(ax, 'on');
    axis(ax, [0 10 1e-4 1.2]);

    set(ax, 'FontName', 'Times New Roman');
    set(ax, 'FontSize', fontsize);
    set(ax, 'LineWidth', 1);
    set(ax, 'GridLineStyle', '--');
    set(ax, 'GridAlpha', 0.2);
    set(ax, 'Layer', 'bottom');
    set(ax, 'TickDir', 'in');
    set(ax, 'TickLength', [0.010, 0.010]);

    if isprop(ax, 'GridLineWidth')
        set(ax, 'GridLineWidth', 0.5);
    end

    title(ax, ['Subfigure ', num2str(idx)], ...
        'FontSize', title_fontsize, ...
        'FontName', 'Times New Roman', ...
        'Interpreter', 'latex');

    if idx == 1
        h_legend = legend(ax, 'Method 1', 'Method 2', ...
            'FontSize', legendfontsize, ...
            'FontWeight', 'normal', ...
            'Location', 'southwest', ...
            'Interpreter', 'latex');
        set(h_legend, 'FontName', 'Times New Roman');
    end
end

hx = xlabel(t, 'SNR(dB)', ...
    'FontSize', label_fontsize, ...
    'FontName', 'Times New Roman', ...
    'Interpreter', 'latex');

hy = ylabel(t, 'WER', ...
    'FontSize', label_fontsize, ...
    'FontName', 'Times New Roman', ...
    'Interpreter', 'latex');

drawnow;

all_ax = findall(fig, 'Type', 'axes');
for idx_ax = 1:length(all_ax)
    if isprop(all_ax(idx_ax), 'Toolbar')
        all_ax(idx_ax).Toolbar.Visible = 'off';
    end
end

drawnow;

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, width, height]);
set(fig, 'PaperSize', [width, height]);
set(fig, 'PaperPositionMode', 'manual');

print(fig, 'Fig_2x4.pdf', '-dpdf', '-painters');
print(fig, 'Fig_2x4.eps', '-depsc', '-painters');