clc;
clear all;
close all;
addpath('./functions');

rng('default');

N = 10;
d = 0.5;
lambda = 2 * d;
Pt = 10;

% ULA 一维阵元位置，单位为 wavelength
pos = ((0:N-1) - (N-1) / 2) * d;
% pos = (0:N-1) * d;
normalizedPos = pos / lambda;

% Targets of interest
theta_est = [-40 0 40];

theta_grid = linspace(-90, 90, 200);
beamwidth = 10;

% Desired beam pattern
P_des = zeros(size(theta_grid));
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i) - beamwidth / 2 & theta_grid <= theta_est(i) + beamwidth / 2;
end
P_des(idx) = 1;

figure(1);
plot(theta_grid, P_des, 'LineWidth', 2);
xlabel('Azimuth (deg)');
ylabel('Desired Beam Pattern');
title('Desired Beam Pattern');
grid on;

A = steeringMatrixULA1D(normalizedPos, theta_grid);
%% A. Squared Error Optimization
Rmmse = helperMMSECovariance(normalizedPos, P_des, theta_grid);
Rmmse = Rmmse * (Pt/N);
P_mmse = abs(diag(A'*Rmmse*A))/(4*pi);

fprintf('trace(Rmmse) = %.6f\n',  trace(Rmmse));

%% A. Squared Error Optimization, 不用 cos(theta) 权重，不做积分归一化，不用 barrier/Newton，直接 CVX 最小化二范数。
[Rmmse1, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt);
P_mmse1 = abs(diag(A'*Rmmse1*A))/(4*pi);
fprintf('trace(Rmmse1) = %.6f\n',  trace(Rmmse1));

%% B. Maximum Error Optimization
Rminmax = helperMinMaxCovariance(normalizedPos, P_des, theta_grid);
Rminmax = Rminmax * (Pt/N);
P_minmax = abs(diag(A'*Rminmax*A))/(4*pi);

fprintf('trace(Rminmax) = %.6f\n',  trace(Rminmax));

%% Plot Fig
figure(1);
plot(theta_grid, 10 * log10(P_des / max(P_des) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(theta_grid, 10 * log10(P_mmse / max(P_mmse) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
plot(theta_grid, 10 * log10(P_mmse1 / max(P_mmse1) + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'c'); hold on;
plot(theta_grid, 10 * log10(P_minmax / max(P_minmax) + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');
xlabel('Azimuth (deg)');
ylabel('Normalized (dB)');
legend('Desired', 'SquaredErrorOptimization', 'myMMSE', 'MaximumErrorOptimization');
ylim([-40 5]);
title('Transmit Beam Pattern');
grid on;


%% 

%===========================================
width = 8;%设置图宽，这个不用改
height = 6;%设置图高，这个不用改
fontsize = 18;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
%%========================================================================================
%    开始画图
%%========================================================================================

figure(2);
% fig(h, 'units','inches','width',width, 'height', height, 'font','Times New Roman','fontsize',fontsize);%这是用于裁剪figure的。需要把fig.m文件放在一个文件夹中

%颜色集合，这是默认的八种颜色，颜色的数量可以更改0.101785714285714 0.1699604743083 0.380357142857143 0.798418972332016
ColorSet = [
0       0       0   % 黑色
1       0       0      % 红色
0       0       1      % 蓝色
0       0.5     0   % 深绿色
1       0.75    0.8 % 粉色
0.5     0       0.5  % 紫色
0.5     0.5     0   % 橄榄绿
0       0       0.5 % 深蓝色
0       1       0      % 绿色
1       0       1      % 洋红色
0       1       1      % 青色
1       0.5     0      % 橙色
0.6     0.3     0   % 棕色
0       0       0
0.5     0       0   % 深红色
0.5     0.5     0.5 % 灰色
    ];
%设置循环使用的颜色集合
set(gcf, 'DefaultAxesColorOrder', ColorSet);
% gca表示对axes的设置；  gcf表示对figure的设置
set(gcf, 'Units', 'inches');
% set(gcf, 'Position', [0, 0, width, height]);
set(gcf, 'Color', 'white'); % 设置背景是白色的 原先是灰色的 论文里面不好看
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
set(gcf, 'PaperPositionMode', 'manual');

P_des_plot = Pt * P_des / (2 * pi * trapz(deg2rad(theta_grid), P_des .* cosd(theta_grid)));
plot(theta_grid, 10 * log10(P_des_plot + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'k'); hold on;
plot(theta_grid, 10 * log10(P_mmse + eps), 'LineStyle', '--', 'LineWidth', 2, 'Color', 'r'); hold on;
p1 = plot(theta_grid, 10 * log10(P_mmse1 + eps), 'LineStyle', '--', 'LineWidth', 2); hold on;
p1.Color = '#43c951';
plot(theta_grid, 10 * log10(P_minmax + eps), 'LineStyle', '-', 'LineWidth', 2, 'Color', 'b');

%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

h_legend = legend('Desired', 'SquaredErrorOptimization', 'ProposedSquaredErrorOptim', 'MaximumErrorOptimization', 'Interpreter','latex');  %图例，与上面的曲线先后对应

legendsize = 14;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','NorthEast');
set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 18;
xlabel('$\theta^{\circ}$','FontName','Times New Roman','FontSize',labelsize,'FontWeight','normal','Color','k','Interpreter','latex');%横坐标标号,坐标轴label字体、字体大小
ylabel('Beampattern (dB)','FontName','Times New Roman','FontSize',labelsize,'FontWeight','normal','Color','k','Interpreter','latex');%纵坐标标号，坐标轴label字体、字体大小

axis([-90 90 -40 10]);         % 横纵坐标范围

%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_3_2.pdf', '-dpdf', '-vector');


