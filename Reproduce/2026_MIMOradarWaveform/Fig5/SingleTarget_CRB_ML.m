


clc;
clear;
close all;

addpath('./functions');
rng(42);
%% Parameters
M = 10;
L = 1;
d_lambda = 0.5;
theta_true_deg = 0;
alpha = 1;
SNR_dB_vec = -30:2:10;
MC_trials = 1000;
N_coherent = L;
N_orth = L;
beta_coherent = 0.5;
beta_orth = 0;
theta_search_min = theta_true_deg-20;
theta_search_max = theta_true_deg+20;
coarse_step = 0.01;
fine_step = 0.001;
fine_width = 0.01;
%   n = (0:M-1).';
n = (-(M-1)/2 : (M-1)/2).';
a_fun = @(theta_deg) exp(-1j * 2 * pi * d_lambda * n * sind(theta_deg));
da_fun = @(theta_deg) -1j * 2 * pi * d_lambda * cosd(theta_deg) * n .* a_fun(theta_deg);
A_fun = @(theta_deg) a_fun(theta_deg) * a_fun(theta_deg)';
dA_fun = @(theta_deg) da_fun(theta_deg) * a_fun(theta_deg)' + a_fun(theta_deg) * da_fun(theta_deg)';
Rs_fun = @(beta) (1 - beta) * eye(M) + beta * ones(M);

%% CRB and ML
CRB_coherent_deg = zeros(size(SNR_dB_vec));
CRB_orth_deg = zeros(size(SNR_dB_vec));
CRB_coherent_63_deg = zeros(size(SNR_dB_vec));
CRB_orth_63_deg = zeros(size(SNR_dB_vec));
CRB_coherent_67_deg = zeros(size(SNR_dB_vec));
CRB_orth_67_deg = zeros(size(SNR_dB_vec));

RMSE_coherent_deg = zeros(size(SNR_dB_vec));
RMSE_orth_deg = zeros(size(SNR_dB_vec));
for idx = 1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(idx);
    SNR_lin = 10^(SNR_dB / 10);
    sigma2 = abs(alpha)^2 / SNR_lin;

    a0 = a_fun(theta_true_deg);
    adot0 = da_fun(theta_true_deg);
    Rs_coherent =  a_fun(theta_true_deg) * a_fun(theta_true_deg)';  %  Rs_fun(beta_coherent);
    Rs_orth = Rs_fun(beta_orth);

    % Eq.(57-59)
    CRB_coherent_rad2 = crb_single_target_equiv_model(theta_true_deg, alpha, Rs_coherent, M, N_coherent, sigma2, A_fun, dA_fun);
    CRB_orth_rad2 = crb_single_target_equiv_model(theta_true_deg, alpha, Rs_orth, M, N_orth, sigma2, A_fun, dA_fun);
    CRB_coherent_deg(idx) = sqrt(max(real(CRB_coherent_rad2), 0)) * 180 / pi;
    CRB_orth_deg(idx) = sqrt(max(real(CRB_orth_rad2), 0)) * 180 / pi;
    
    SNR_coherent_eff = N_coherent * abs(alpha)^2 / sigma2;
    SNR_orth_eff = N_orth * abs(alpha)^2 / sigma2; 
    
    % Eq.(63)
    CRB_orth_63_rad2 = crb_single_63(a0, adot0, Rs_orth, SNR_orth_eff);
    CRB_orth_63_deg(idx) = sqrt(max(real(CRB_orth_63_rad2), 0)) * 180 / pi;
    CRB_coherent_63_rad2 = crb_single_63(a0, adot0, Rs_coherent, SNR_coherent_eff);
    CRB_coherent_63_deg(idx) = sqrt(max(real(CRB_coherent_63_rad2), 0)) * 180 / pi;

    % Eq.(67)
    CRB_orth_67_rad2 = crb_single_67_correct(a0, adot0, Rs_orth, SNR_orth_eff); 
    CRB_orth_67_deg(idx) = sqrt(max(real(CRB_orth_67_rad2), 0)) * 180 / pi;
    CRB_coherent_67_rad2 = crb_single_67_correct(a0, adot0, Rs_coherent, SNR_coherent_eff);
    CRB_coherent_67_deg(idx) = sqrt(max(real(CRB_coherent_67_rad2), 0)) * 180 / pi;

    theta_est_coherent = zeros(MC_trials, 1);
    theta_est_orth = zeros(MC_trials, 1);
    fprintf('SNR = %.1f dB\n', SNR_dB);
    for mc = 1:MC_trials
        z_coherent = simulate_single_target_equiv(theta_true_deg, alpha, Rs_coherent, M, N_coherent, sigma2, A_fun);
        z_orth = simulate_single_target_equiv(theta_true_deg, alpha, Rs_orth, M, N_orth, sigma2, A_fun);
        theta_est_coherent(mc) = ml_single_target_search(z_coherent, Rs_coherent, M, N_coherent, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
        theta_est_orth(mc) = ml_single_target_search(z_orth, Rs_orth, M, N_orth, A_fun, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width);
    end
    RMSE_coherent_deg(idx) = sqrt(mean((theta_est_coherent - theta_true_deg).^2));
    RMSE_orth_deg(idx) = sqrt(mean((theta_est_orth - theta_true_deg).^2));
end
%% Plot

figure(1);
semilogy(SNR_dB_vec, CRB_orth_deg, 'k-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_orth_63_deg, 'ro', 'MarkerSize', 7, 'LineWidth', 1.3); hold on;
semilogy(SNR_dB_vec, CRB_orth_67_deg, 'b^', 'MarkerSize', 7, 'LineWidth', 1.3);
xlabel('SNR [dB]');
ylabel('RMSE CRB of \theta [deg]');
legend('Equivalent FIM', 'Formula (63)', 'Corrected Formula (67)', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Orthogonal signal: comparison of three CRB calculations');
%% CRB comparison, coherent
figure(2);
semilogy(SNR_dB_vec, CRB_coherent_deg, 'k-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_coherent_63_deg, 'ro', 'MarkerSize', 7, 'LineWidth', 1.3); hold on;
semilogy(SNR_dB_vec, CRB_coherent_67_deg, 'b^', 'MarkerSize', 7, 'LineWidth', 1.3);
xlabel('SNR [dB]');
ylabel('RMSE CRB of \theta [deg]');
legend('Equivalent FIM', 'Formula (63)', 'Corrected Formula (67)', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Coherent signal: comparison of three CRB calculations');

figure(3);
semilogy(SNR_dB_vec, CRB_coherent_deg, 'b-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_orth_deg, 'r-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, RMSE_coherent_deg, 'b--', 'MarkerSize', 7, 'LineWidth', 1); hold on;
semilogy(SNR_dB_vec, RMSE_orth_deg, 'r--', 'MarkerSize', 7, 'LineWidth', 1);
xlabel('SNR [dB]');
ylabel('RMSE of \theta [deg]');
legend('CRB, \beta = 1', 'CRB, \beta = 0', 'ML, \beta = 1', 'ML, \beta = 0', 'Location', 'southwest');
grid on;
grid minor;
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
title('Single-target CRB and ML performance versus SNR');



%% ===========================================
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

figure(4);
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

semilogy(SNR_dB_vec, CRB_coherent_deg, 'b-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, CRB_orth_deg, 'r-', 'LineWidth', 1.8); hold on;
semilogy(SNR_dB_vec, RMSE_coherent_deg, 'b--o', 'MarkerSize', 7, 'LineWidth', 1); hold on;
semilogy(SNR_dB_vec, RMSE_orth_deg, 'r--d', 'MarkerSize', 7, 'LineWidth', 1);
%-------------------------------------------------------------------

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');
h_legend =  legend('CRB, coherent signal', 'CRB, orthogonal signal', 'ML, coherent signal', 'ML, orthogonal signal', 'Interpreter', 'latex');
legendsize = 14;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1, 'Location', 'northeast');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 14;

xlabel('SNR (dB)', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('RMSE of $\theta^{\circ}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% xlim([-90, 90]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

print(gcf, 'Fig_5_2.pdf', '-dpdf', '-vector');



