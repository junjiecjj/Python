clc;
clear all;
close all;
addpath('./functions_2007TSP_OnProb');
rng(42);

%% Fig.4(c): Minimum sidelobe design with PAR < 1.2
M = 10;
L = 256;
rho = 1.1;

C = 1;
c = ones(M, 1) * C / M;

theta0 = 0;
theta1 = -10;
theta2 = 10;
theta_null = -30;
null_level_dB = -40;

Omega = [-90:0.1:-20, 20:0.1:90];
theta_plot = -90:0.1:90;

a = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));

%% Step 1：设计带 null 约束的 minimum sidelobe optimal R
R_opt = MinimumSidelobeDesignWithNull(c, M, theta0, theta1, theta2, Omega, theta_null, null_level_dB);

%% Step 2：CA 合成波形
X_optR = WaveformSynthesisXoptimR(R_opt, L, rho);
X_par = WaveformSynthesisXwithPAR(R_opt, L, rho);

%% Step 3：计算样本协方差矩阵
R_optR = X_optR * X_optR' / L;
R_par = X_par * X_par' / L;

%% Step 4：计算 beampattern
P_R = zeros(size(theta_plot));
P_optR = zeros(size(theta_plot));
P_par = zeros(size(theta_plot));

for i = 1:length(theta_plot)
    ai = a(theta_plot(i));
    P_R(i) = real(ai' * R_opt * ai);
    P_optR(i) = real(ai' * R_optR * ai);
    P_par(i) = real(ai' * R_par * ai);
end
% 
% P_R(P_R < 0) = 0;
% P_optR(P_optR < 0) = 0;
% P_par(P_par < 0) = 0;

P_R_dB = 10 * log10(P_R);
P_optR_dB = 10 * log10(P_optR);
P_par_dB = 10 * log10(P_par);

%% Step 5：输出检查量
[~, idx_null] = min(abs(theta_plot - theta_null));

err_optR = norm(R_optR - R_opt, 'fro') / norm(R_opt, 'fro');
err_par = norm(R_par - R_opt, 'fro') / norm(R_opt, 'fro');

fprintf('CA optimal R: R error = %.4e, %.4f dB\n', err_optR, 20 * log10(err_optR));
fprintf('CA PAR < %.1f: R error = %.4e, %.4f dB\n', rho, err_par, 20 * log10(err_par));

par_optR = L * max(abs(X_optR).^2, [], 2) ./ sum(abs(X_optR).^2, 2);
par_par = L * max(abs(X_par).^2, [], 2) ./ sum(abs(X_par).^2, 2);

fprintf('CA optimal R max PAR = %.4f\n', max(par_optR));
fprintf('CA PAR < %.1f max PAR = %.4f\n', rho, max(par_par));

fprintf('Optimal R null at %.1f° = %.4f dB\n', theta_null, P_R_dB(idx_null));
fprintf('CA optimal R null at %.1f° = %.4f dB\n', theta_null, P_optR_dB(idx_null));
fprintf('CA PAR < %.1f null at %.1f° = %.4f dB\n', rho, theta_null, P_par_dB(idx_null));

%% Step 6：画 Fig.4(c)
% figure(1);
% plot(theta_plot, P_R_dB, 'k-', 'LineWidth', 1.5); hold on;
% plot(theta_plot, P_optR_dB, 'b-.', 'LineWidth', 1.5); hold on;
% plot(theta_plot, P_par_dB, 'r--', 'LineWidth', 1.5); hold on;
% xline(theta_null, 'k:', 'LineWidth', 2);
% 
% grid on;
% xlabel('\theta (degrees)');
% ylabel('Normalized Power (dB)');
% legend('Optimal R', 'CA: optimal R', 'CA: PAR < 1.2', 'Location', 'best');
% title('Fig. 4(c): Minimum Sidelobe Design, PAR < 1.2');
% xlim([-90, 90]);
% % ylim([-60, 5]);

%%===========================================
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
set(gcf, 'PaperPositionMode', 'manual');

p1 = plot(theta_plot, P_R_dB, '-', 'LineWidth', 2); hold on;
p1.Color = '#A9A9A9';

plot(theta_plot, P_optR_dB, 'b--', 'LineWidth', 1.5); hold on;
plot(theta_plot, P_par_dB, 'r:', 'LineWidth', 1.5); hold on;
xline(theta_null, 'k:', 'LineWidth', 2);
%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');
h_legend =  legend('Optimized R', 'CA: strict R', 'CA: PAR $ \leq $ 1.1', 'null','Interpreter', 'latex');
legendsize = 14;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','NorthEast');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 14;

xlabel('$\theta^{\circ}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel("Beampattern (dB)", 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
xlim([-90, 90]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_4_4b.pdf', '-dpdf', '-vector');
