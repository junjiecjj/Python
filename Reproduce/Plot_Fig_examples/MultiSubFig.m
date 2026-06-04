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

xlabel_ypos = -0.18; % xlabel 离坐标轴/刻度文字的距离，越负越远
ylabel_xpos = -0.20; % ylabel 离坐标轴/刻度文字的距离，越负越远
tick_length = 0.010; % 刻度线长度，间接影响刻度文字离坐标轴的距离

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');


fig = figure(1);
set(fig, 'Units', 'inches');
set(fig, 'Position', [1, 1, width, height]);
set(fig, 'Color', 'w');
set(fig, 'Renderer', 'painters');
rows = 1;
cols = 4;
t = tiledlayout(fig, rows, cols);
t.TileSpacing = 'tight';
t.Padding = 'tight';
ax_list = gobjects(rows * cols, 1);
hx_list = gobjects(rows * cols, 1);
hy_list = gobjects(rows * cols, 1);


x = 0:0.1:10;

for idx = 1:cols*rows
    ax = nexttile(t); ax.Toolbar.Visible = 'off';
    ax.XAxis.TickLabelGapOffset = 0.01;
    ax.YAxis.TickLabelGapOffset = 0.01;
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
    set(hx, 'VerticalAlignment', 'cap');   % 使标签紧贴轴线
    hy = ylabel(ax, 'WER', 'FontSize', ylabel_fontsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    % set(hy, 'VerticalAlignment', 'cap');   % 使标签紧贴轴线
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

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, width, height]);
set(fig, 'PaperSize', [width, height]);
set(fig, 'PaperPositionMode', 'manual');


% all_ax = findall(fig, 'Type', 'axes');
% for idx_ax = 1:length(all_ax)
%     if isprop(all_ax(idx_ax), 'Toolbar')
%         all_ax(idx_ax).Toolbar.Visible = 'off';
%     end
% end


drawnow;

print(fig, 'MultiSubFig.pdf', '-dpdf', '-vector');
% print(fig, 'MultiSubFig.pdf', '-dpdf', '-painters');
print(fig, 'MultiSubFig.eps', '-depsc', '-vector');
print(fig, 'MultiSubFig.png', '-dpng', '-r300');