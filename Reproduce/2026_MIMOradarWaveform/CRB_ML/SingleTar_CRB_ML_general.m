clc;
clear;
close all;
rng(42);

addpath('./functions');
%% Parameters
M = 10;  % M 收 N 发
N = 8;
L = 128;
d_lambda = 0.5;
Pt = 2;
theta_true_deg = 10;
beta = 1;
SNR_dB_vec = -40:4:0;
MC_trials = 2000;

theta_search_min = theta_true_deg - 20;
theta_search_max = theta_true_deg + 20;
coarse_step = 0.02;
fine_step = 0.001;
fine_width = 0.1;

% 两种形式的CRB对得上得使用这样的居中nr；
nr = (-(M - 1) / 2 : (M - 1) / 2).';
% nr = (0:M-1).';

% 而对nt没要求;
% nt = (-(N - 1) / 2 : (N - 1) / 2).';
nt = (0:N-1).';

a_fun = @(theta_rad) exp(-1j * 2 * pi * d_lambda * nt * sin(theta_rad));
v_fun = @(theta_rad) exp(-1j * 2 * pi * d_lambda * nr * sin(theta_rad));
da_fun = @(theta_rad) -1j * 2 * pi * d_lambda * cos(theta_rad) * nt .* a_fun(theta_rad);
dv_fun = @(theta_rad) -1j * 2 * pi * d_lambda * cos(theta_rad) * nr .* v_fun(theta_rad);

theta_true_rad = theta_true_deg * pi / 180;
a0 = a_fun(theta_true_rad);

R_orth = eye(N);

% G = randn(N, N) + 1j * randn(N, N);
% R_random = G * G';
% R_coherent = (R_random + R_random') / 2;
% 
% cohe = 0.9;
% R_coherent = (1-cohe)*eye(N) + cohe*ones(N);

R_coherent = a0 * a0';
epsilon = 0.3;
R_coherent = (1 - epsilon) * R_coherent + epsilon * eye(N);

R_orth = Pt * R_orth / real(trace(R_orth));
R_coherent = Pt * R_coherent / real(trace(R_coherent));

%% Initialization
CRB_orth_A_deg = zeros(size(SNR_dB_vec));
CRB_coherent_A_deg = zeros(size(SNR_dB_vec));
CRB_orth_a_deg = zeros(size(SNR_dB_vec));
CRB_coherent_a_deg = zeros(size(SNR_dB_vec));

RMSE_orth_deg = zeros(size(SNR_dB_vec));
RMSE_coherent_deg = zeros(size(SNR_dB_vec));

%% Monte Carlo Simulation
for idxSNR = 1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(idxSNR);
    SNR_lin = 10^(SNR_dB / 10);
    sigma2 = Pt * abs(beta)^2 / SNR_lin;

    CRB_orth_A_rad2 = crb_single_target_eqA(theta_true_rad, beta, R_orth, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_coherent_A_rad2 = crb_single_target_eqA(theta_true_rad, beta, R_coherent, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_orth_a_rad2 = crb_single_target_eqa(theta_true_rad, beta, R_orth, L, sigma2, a_fun, v_fun, da_fun, dv_fun);
    CRB_coherent_a_rad2 = crb_single_target_eqa(theta_true_rad, beta, R_coherent, L, sigma2, a_fun, v_fun, da_fun, dv_fun);

    CRB_orth_A_deg(idxSNR) = sqrt(max(real(CRB_orth_A_rad2), 0)) * 180 / pi;
    CRB_coherent_A_deg(idxSNR) = sqrt(max(real(CRB_coherent_A_rad2), 0)) * 180 / pi;
    CRB_orth_a_deg(idxSNR) = sqrt(max(real(CRB_orth_a_rad2), 0)) * 180 / pi;
    CRB_coherent_a_deg(idxSNR) = sqrt(max(real(CRB_coherent_a_rad2), 0)) * 180 / pi;

    theta_est_orth_deg = zeros(MC_trials, 1);
    theta_est_coherent_deg = zeros(MC_trials, 1);

    fprintf('SNR = %.1f dB\n', SNR_dB);

    for idxMC = 1:MC_trials
        X_orth = generate_waveform_exact_covariance(R_orth, L);
        X_coherent = generate_waveform_exact_covariance(R_coherent, L);

        Y_orth = simulate_single_target_matrix(theta_true_rad, beta, X_orth, sigma2, a_fun, v_fun);
        Y_coherent = simulate_single_target_matrix(theta_true_rad, beta, X_coherent, sigma2, a_fun, v_fun);

        theta_est_orth_deg(idxMC) = ml_single_target_matrix_search(Y_orth, X_orth, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width, a_fun, v_fun);
        theta_est_coherent_deg(idxMC) = ml_single_target_matrix_search(Y_coherent, X_coherent, theta_search_min, theta_search_max, coarse_step, fine_step, fine_width, a_fun, v_fun);
    end

    RMSE_orth_deg(idxSNR) = sqrt(mean((theta_est_orth_deg - theta_true_deg).^2));
    RMSE_coherent_deg(idxSNR) = sqrt(mean((theta_est_coherent_deg - theta_true_deg).^2));
end

%% Plot

%% ===========================================
width = 8;%设置图宽，这个不用改
height = 6;%设置图高，这个不用改
fontsize = 14;%设置图中字体大小
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

semilogy(SNR_dB_vec, CRB_coherent_A_deg, 'b-', 'LineWidth', 1.8);  hold on;
semilogy(SNR_dB_vec, CRB_orth_A_deg, 'r-', 'LineWidth', 1.8);
semilogy(SNR_dB_vec, CRB_coherent_a_deg, 'bo', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, CRB_orth_a_deg, 'rd', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, RMSE_coherent_deg, 'b--', 'LineWidth', 1.3, 'MarkerSize', 7);
semilogy(SNR_dB_vec, RMSE_orth_deg, 'r--', 'LineWidth', 1.3, 'MarkerSize', 7);

h_legend = legend('CRB Eq. (80), coherent signal', ...
       'CRB Eq. (80), orthogonal signal', ...
       'CRB Eq. (81), coherent signal', ...
       'CRB Eq. (81), orthogonal signal', ...
       'ML, coherent signal', ...
       'ML, orthogonal signal', ...
       'Location', 'northeast', ...
       'Interpreter', 'latex');

legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','best');
labelsize = 14;

xlabel('SNR (dB)', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('RMSE of $\theta^{\circ}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
% xlim([min(SNRdB), max(SNRdB)]); 
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');
set(get(gca, 'XAxis'), 'FontSize', 12);  % 调整坐标轴刻度标签（tick labels）的字体大小
set(get(gca, 'YAxis'), 'FontSize', 12);
%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.1, 0.1, 0.87, 0.86]);

print(gcf, 'Fig_5_1a.pdf', '-dpdf', '-vector');

