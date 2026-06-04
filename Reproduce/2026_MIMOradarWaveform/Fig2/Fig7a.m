


%% 问题(19)的SOCP求解 
clc;
clear all;
close all;

rng(42); 
addpath('./functions');

%% 1. 参数设置（示例，可修改）
M = 10;                     % 天线数
C = 1;
c = ones(M, 1) * C / M;     % 对角元固定值          
theta_est = [0];            % 目标角度估计（度）

K = length(theta_est);      % 目标个数
a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

Delta = 30;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - Delta & theta_grid <= theta_est(i) + Delta;
end
P_des(idx) = 1;
L = length(theta_grid);

% 权重
w_l = ones(L, 1);           % 所有网格点权重相同

wc = 0;
[R_opt0, alpha0, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);
wc = 1;
[R_opt1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, wc, theta_est, theta_grid, P_des);

p_des = abs(P_des * alpha0+eps);
fprintf('trace(R_opt0) = %.6f\n',  trace(R_opt0));
 
P_opt0 = zeros(size(theta_grid));
P_opt1 = zeros(size(theta_grid));
for i = 1:length(theta_grid)
    a_theta = a(theta_grid(i));
    P_opt0(i) = real(a_theta' * R_opt0 * a_theta);
    P_opt1(i) = real(a_theta' * R_opt1 * a_theta);
end

%% 可选：绘制发射波束图对比
figure(1);
plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt0, 'r-', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt1, 'b--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, pow2db(p_des/max(p_des)), 'k--', 'LineWidth', 1.5); hold on;
% plot(theta_grid, pow2db(P_opt0/max(P_opt0)), 'r--', 'LineWidth', 1.5); hold on;
% ylim([-30 2]);

xlabel('\theta (degrees)');
ylabel('Beampattern');
legend('Desired',  'Optimized,w_c=0', 'Optimized,w_c=1');
title('Transmit Beampattern');
grid on;



%% 

%===========================================
width = 8;%设置图宽，这个不用改
height = 6;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。
%%========================================================================================
%    开始画图
%%========================================================================================

figure(2);
% fig(h, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中

% gca表示对axes的设置；  gcf表示对figure的设置
set(gcf, 'Units', 'inches');
% set(gcf, 'Position', [0, 0, width, height]);
set(gcf, 'Color', 'white'); % 设置背景是白色的 原先是灰色的 论文里面不好看
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
set(gcf, 'PaperPositionMode', 'manual');
 
plot(theta_grid, abs(P_des * alpha0), 'k--', 'LineWidth', 1.5); hold on;
plot(theta_grid, P_opt0, 'b-', 'LineWidth', 1.5); hold on; hold on;
plot(theta_grid, P_opt1, 'r--', 'LineWidth', 1.5); hold on;

%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

h_legend = legend('Desired', 'BeamMatch, $w_c$=0', 'BeamMatch, $w_c$=1', 'Interpreter','latex');  %图例，与上面的曲线先后对应

legendsize = 12;
set(h_legend,'FontName','宋体','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','NorthEast');
set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 18;
xlabel('$\theta^{\circ}$','FontName','Times New Roman','FontSize',labelsize,'FontWeight','normal','Color','k','Interpreter','latex');%横坐标标号,坐标轴label字体、字体大小
ylabel('Transmit Beampattern','FontName','Times New Roman','FontSize',labelsize,'FontWeight','normal','Color','k','Interpreter','latex');%纵坐标标号，坐标轴label字体、字体大小

% axis([0 2.5 1e-7 1]);         % 横纵坐标范围

%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_2_3.pdf', '-dpdf', '-vector');






