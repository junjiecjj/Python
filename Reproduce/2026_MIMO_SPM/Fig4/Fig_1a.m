

clc;
clear all;
close all;

rng(42); 
addpath('./functions_2007TSP_OnProb');


%% 问题(19)的SOCP求解, in "2007-TSP-On Probing Signal Design For MIMO Radar"
N = 10;                       % 天线数
c = ones(N, 1);                % 对角元固定值
theta_est = [0];   % 目标角度估计（度）

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

% 权重
w_l = ones(L, 1);           % 所有网格点权重相同
wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, N, w_l, wc, theta_est, theta_grid, P_des);
p_des = abs(P_des * alpha0+eps);

P_opt0 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
end

rho = 1;

%%  Optimal R in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
L  = 256;
X_optR = WaveformSynthesisXoptimR(R_opt0, L, rho );

Rhat1 = X_optR * X_optR'/L;
P_opt1 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt1(i) = real(a_theta' * Rhat1 * a_theta);
end

%%  PAR < rho in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"

X_par = WaveformSynthesisXwithPAR(R_opt0, L, rho);
Rhat2 = X_par * X_par'/L;
P_opt2 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt2(i) = real(a_theta' * Rhat2 * a_theta);
end

%% 可选：绘制发射波束图对比

%% ===========================================
width = 6;%设置图宽，这个不用改
height = 4;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
% ===========================================
figure(1);
% fig(h, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中

% gca表示对axes的设置；  gcf表示对figure的设置
set(gcf, 'Units', 'inches');
% set(gcf, 'Position', [0, 0, width, height]);
set(gcf, 'Color', 'white'); % 设置背景是白色的 原先是灰色的 论文里面不好看
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
p2 = plot(theta_grid, P_opt0, 'r-', 'LineWidth', 2); hold on;
p2.Color = '#A9A9A9';
plot(theta_grid, P_opt1, 'r-.', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt2, 'b:', 'LineWidth', 1.5); hold on;

h_legend = legend('Desired', ...
                  'Optimized, $w_c$=0',...
                  'CA:strict R',...
                  'CA:PAR = 1',...
                  'Interpreter', 'latex'...
                  );  %图例，与上面的曲线先后对应

legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1, 'Location','NorthEast');
labelsize = 14;

xlabel('$\theta^{\circ}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel("Beampattern", 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');

%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');
set(get(gca, 'XAxis'), 'FontSize', 12);  % 调整坐标轴刻度标签（tick labels）的字体大小
set(get(gca, 'YAxis'), 'FontSize', 12);
%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.1, 0.1, 0.87, 0.86]);
print(gcf, 'Fig_4_1a.pdf', '-dpdf', '-vector');






























































