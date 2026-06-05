

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

L  = 256;
Rho = [1, 1.1, 2];

PAR1 = zeros(length(Rho), N);

for k = 1:length(Rho)
    rho = Rho(k);

    %%  Optimal R in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
    X_optR = WaveformSynthesisXoptimR(R_opt0, L, rho );

    Rhat1 = X_optR * X_optR'/L;
    P_opt1 = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        a_theta = a(theta_grid(i));
        P_opt1(i) = real(a_theta' * Rhat1 * a_theta);
    end
    
    PAR1(k, :) = max(abs(X_optR).^2, [], 2) ./ (mean(abs(X_optR).^2, 2));

    %%  PAR < rho in "2008-TSP-Waveform Synthesis for Diversity-Based Transmit Beampattern Design"
    X_par = WaveformSynthesisXwithPAR(R_opt0, L, rho  );
    Rhat2 = X_par * X_par'/L;
    P_opt2 = zeros(size(theta_grid));
    for i = 1:length(theta_grid)
        a_theta = a(theta_grid(i));
        P_opt2(i) = real(a_theta' * Rhat2 * a_theta);
    end
    PAR2(k, :) = max(abs(X_par).^2, [], 2) ./ (mean(abs(X_par).^2, 2));
    %% 可选：绘制发射波束图对比
    figure(k);
    plot(theta_grid, p_des, 'k--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt0, 'r-', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt1, 'b--', 'LineWidth', 1.5); hold on;
    plot(theta_grid, P_opt2, 'c--', 'LineWidth', 1.5); hold on;
 
    xlabel('\theta (degrees)');
    ylabel('Beampattern');
    legend('Desired',  'Optimized,w_c=0', 'CA:optimal R', ['CA:PAR = ', num2str(rho)]);
    title('Transmit Beampattern');
    grid on;

end 

% figure(length(Rho) + 1);
% plot(1:1:N, PAR1(1,:), 'r-o', 'LineWidth', 1.5); hold on;
% plot(1:1:N, PAR1(2,:), 'b--*', 'LineWidth', 1.5); hold on;
% plot(1:1:N, PAR1(3,:), 'c--d', 'LineWidth', 1.5); hold on;
% 
% xlabel('Index of Transmit Antenna','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% ylabel('PAR','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% legend('CA(PAR=1):optimal R', 'CA(PAR$\leq$1.1):optimal R', 'CA(PAR$\leq$2):optimal R','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% ylim([-1, 8]);
% grid on;
% 
% figure(length(Rho) + 2);
% plot(1:1:N, PAR2(1,:), 'r-o', 'LineWidth', 1.5); hold on;
% plot(1:1:N, PAR2(2,:), 'b--*', 'LineWidth', 1.5); hold on;
% plot(1:1:N, PAR2(3,:), 'c--d', 'LineWidth', 1.5); hold on;
% 
% xlabel('Index of Transmit Antenna','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% ylabel('PAR','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% legend('CA(PAR=1):optimal R', 'CA(PAR$\leq$1.1):optimal R', 'CA(PAR$\leq$2):optimal R','FontSize', 12, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% ylim([-1, 8]);
% grid on;

%===========================================
width = 6;%设置图宽，这个不用改
height = 4;%设置图高，这个不用改
fontsize = 14;%设置图中字体大小
linewidth = 2;%设置线宽，一般大小为2，好看些。1是默认大小
markersize = 10;%标记的大小，按照个人喜好设置。
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');
%%========================================================================================
%         开始画图
%%========================================================================================
%%
figure(length(Rho) + 1);
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

plot(1:1:N, PAR1(1,:), 'k-o', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR1(2,:), 'r--*', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR1(3,:), 'b--d', 'LineWidth', 1.5); hold on;

%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');
h_legend =  legend('CA(PAR=1):optimal R', 'CA(PAR$\leq$1.1):optimal R', 'CA(PAR$\leq$2):optimal R', 'Interpreter', 'latex');
legendsize = 14;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','NorthEast');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 14;

xlabel('Index of Transmit Antenna','FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('PAR','FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylim([-1, 8]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_4_2.pdf', '-dpdf', '-vector');


%%
figure(length(Rho) + 2);
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

plot(1:1:N, PAR2(1,:), 'k-o', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR2(2,:), 'r--*', 'LineWidth', 1.5); hold on;
plot(1:1:N, PAR2(3,:), 'b--d', 'LineWidth', 1.5); hold on;

%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');
h_legend =  legend('CA(PAR=1):optimal R', 'CA(PAR$\leq$1.1):optimal R', 'CA(PAR$\leq$2):optimal R', 'Interpreter', 'latex');
legendsize = 14;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','NorthEast');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 14;

xlabel('Index of Transmit Antenna','FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('PAR','FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylim([-1, 8]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_4_3.pdf', '-dpdf', '-vector');






















