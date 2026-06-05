

clc;
clear all;
close all;

rng(42); 
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');
%% 
N = 10;                        % 天线数
c = ones(N, 1)/N;              % 对角元固定值
theta_est = [0];               % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:N-1)' * sind(theta));  % M×1

Delta = 30;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i)-Delta & theta_grid <= theta_est(i)+Delta;
end
P_des(idx) = 1;
L = length(theta_grid);

%% 问题(19)的SOCP求解, in "2007-TSP-On Probing Signal Design For MIMO Radar"
w_l = ones(L, 1);           % 所有网格点权重相同
wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, N, w_l, wc, theta_est, theta_grid, P_des);
p_des = abs(P_des * alpha0+eps);


%  Optimal R in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
rho = 1;
L  = 256;
X_optR = WaveformSynthesisXoptimR(R_opt0, L,  rho );
Rhat_pro1_optimR = X_optR * X_optR'/L;
% PAR < rho in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
X_par = WaveformSynthesisXwithPAR(R_opt0, L, rho);
Rhat_pro1_PAR = X_par * X_par'/L;

rho = 1.1;
L  = 256;
X_optR = WaveformSynthesisXoptimR(R_opt0, L,  rho );
Rhat_pro11_optimR = X_optR * X_optR'/L;
X_par = WaveformSynthesisXwithPAR(R_opt0, L, rho);
Rhat_pro11_PAR = X_par * X_par'/L;

rho = 2;
L  = 256;
X_optR = WaveformSynthesisXoptimR(R_opt0, L,  rho );
Rhat_pro2_optimR = X_optR * X_optR'/L;
X_par = WaveformSynthesisXwithPAR(R_opt0, L, rho);
Rhat_pro2_PAR = X_par * X_par'/L;

P_opt0 = zeros(size(theta_grid));
P_pro1_optimR = zeros(size(theta_grid));
P_pro1_PAR = zeros(size(theta_grid));
P_pro11_optimR = zeros(size(theta_grid));
P_pro11_PAR = zeros(size(theta_grid));
P_pro2_optimR = zeros(size(theta_grid));
P_pro2_PAR = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
    P_pro1_optimR(i) = real(a_theta' * Rhat_pro1_optimR * a_theta);
    P_pro1_PAR(i) = real(a_theta' * Rhat_pro1_PAR * a_theta);
    P_pro11_optimR(i) = real(a_theta' * Rhat_pro11_optimR * a_theta);
    P_pro11_PAR(i) = real(a_theta' * Rhat_pro11_PAR * a_theta);
    P_pro2_optimR(i) = real(a_theta' * Rhat_pro2_optimR * a_theta);
    P_pro2_PAR(i) = real(a_theta' * Rhat_pro2_PAR * a_theta);
end
% 
% % 绘制发射波束图对比
% figure(1);
% plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, P_opt0, 'r-', 'LineWidth', 1.5); hold on;
% plot(theta_grid, P_opt1, 'b--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, P_opt2, 'c--', 'LineWidth', 1.5); hold on;
% 
% xlabel('\theta (degrees)');
% ylabel('Beampattern');
% legend('Desired',  'Optimized,w_c=0', 'CA:optimal R', 'CA:PAR = 1.1');
% title('Transmit Beampattern');
% grid on;


%% ========== IEEE-style 1x5 绘图，不含 MUSIC，稳定保存 PDF ==========

width = 15;
height = 4;
fontsize = 14;
linewidth = 1;
markersize = 6;
legendfontsize  = 10;
xlabel_fontsize = 16;
ylabel_fontsize = 16;
title_fontsize = 10;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

fig = figure(2);
set(fig, 'Units', 'inches');
set(fig, 'Position', [1, 1, width, height]);
set(fig, 'Color', 'w');
set(fig, 'Renderer', 'painters');
rows = 1;
cols = 3;
t = tiledlayout(fig, rows, cols);
t.TileSpacing = 'tight';
t.Padding = 'tight';
ax_list = gobjects(rows * cols, 1);
hx_list = gobjects(rows * cols, 1);
hy_list = gobjects(rows * cols, 1);

blueColor = [0, 0.4470, 0.7410];
specData = {P_pro1_optimR, P_pro1_PAR, P_pro11_optimR, P_pro11_PAR, P_pro2_optimR, P_pro2_PAR};
specName = {'CA:optimal R', 'CA:PAR = 1', 'CA:optimal R', 'CA:PAR = 1.1', 'CA:optimal R', 'CA:PAR = 2'};

for ii = 1:3
    ax = nexttile(t); ax.Toolbar.Visible = 'off';
    ax.XAxis.TickLabelGapOffset = 0.01;
    ax.YAxis.TickLabelGapOffset = 0.01;

    yData1 = specData{2*ii-1};
    yData2 = specData{2*ii};
    plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold(ax, 'on');
    p1 = plot(theta_grid, P_opt0, '-', 'LineWidth', 1.5); hold(ax, 'on');
    p1.Color = '#A9A9A9';
    plot(ax, theta_grid, yData1, 'r-.', 'LineWidth', 1.5);  hold(ax, 'on');
    p2 = plot(ax, theta_grid, yData2, 'b:', 'LineWidth', 1.5);
    % p2.Color = '#00841a';

    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', fontsize, 'LineWidth', 1.2);
    set(ax,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer', 'bottom');

    hx = xlabel(ax, '$\theta^{\circ}$', 'FontSize', xlabel_fontsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    set(hx, 'VerticalAlignment', 'cap');   % 使标签紧贴轴线
    hy = ylabel(ax, "Beampattern", 'FontSize', ylabel_fontsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    h_legend = legend(ax, 'Desired','Optimized, $w_c$=0', specName{2*ii-1}, specName{2*ii}, 'FontSize',8, 'FontWeight','normal', 'Location', 'northeast', 'Interpreter', 'latex');
end

% % save Fig
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPosition', [0, 0, width, height]);
set(fig, 'PaperSize', [width, height]);
set(fig, 'PaperPositionMode', 'manual');

drawnow;
print(fig, 'Fig_4_1.pdf', '-dpdf', '-vector');

