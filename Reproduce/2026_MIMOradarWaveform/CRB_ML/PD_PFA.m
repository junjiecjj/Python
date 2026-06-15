
clc;
clear;
close all;
rng(42);

%% Parameters
M = 10;
N = 8;
L = 128;
d_lambda = 0.5;
Pt = 1;
theta_true_deg = 10;
beta = 1;

SNR_dB_vec = -40:1:0;
PFA_vec = logspace(-5, 0, 200);

SNR_fixed_dB = -40;
PFA_fixed = 1e-6;

%% Steering vectors
nr = (-(M - 1) / 2 : (M - 1) / 2).';
nt = (0 : N - 1).';

a_fun = @(theta_rad) exp(-1j * 2 * pi * d_lambda * nt * sin(theta_rad));
v_fun = @(theta_rad) exp(-1j * 2 * pi * d_lambda * nr * sin(theta_rad));

theta_true_rad = theta_true_deg * pi / 180;
a0 = a_fun(theta_true_rad);
v0 = v_fun(theta_true_rad);

%% Transmit covariance matrices
R_orth = eye(N);
R_coherent = a0 * a0';
% cohe = 0.9;
% R_coherent = (1-cohe)*eye(N) + cohe*ones(N);

% G = randn(N, N) + 1j * randn(N, N);
% R_random = G * G';
% R_coherent = (R_random + R_random') / 2;

R_orth = Pt * R_orth / real(trace(R_orth));
R_coherent = Pt * R_coherent / real(trace(R_coherent));

%% Precompute useful terms in rho
v_norm_sq = real(v0' * v0);
gain_orth = real(a0' * R_orth * a0);
gain_coherent = real(a0' * R_coherent * a0);

%% =========================================================
%% Figure 1: P_D versus P_FA for a fixed noise power sigma_s^2
%% =========================================================
SNR_fixed_lin = 10^(SNR_fixed_dB / 10);
sigma2_fixed = Pt * abs(beta)^2 / SNR_fixed_lin;

rho_orth_fixed = abs(beta)^2 * L * v_norm_sq * gain_orth / sigma2_fixed;
rho_coherent_fixed = abs(beta)^2 * L * v_norm_sq * gain_coherent / sigma2_fixed;

eta_vec = chi2inv(1 - PFA_vec, 2);

PD_orth_vs_PFA = 1 - ncx2cdf(eta_vec, 2, rho_orth_fixed);
PD_coherent_vs_PFA = 1 - ncx2cdf(eta_vec, 2, rho_coherent_fixed);

%% =========================================================
%% Figure 2: P_D versus SNR for a fixed P_FA
%% =========================================================
eta_fixed = chi2inv(1 - PFA_fixed, 2);

PD_orth_vs_SNR = zeros(size(SNR_dB_vec));
PD_coherent_vs_SNR = zeros(size(SNR_dB_vec));

for idx = 1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(idx);
    SNR_lin = 10^(SNR_dB / 10);
    sigma2 = Pt * abs(beta)^2 / SNR_lin;

    rho_orth = abs(beta)^2 * L * v_norm_sq * gain_orth / sigma2;
    rho_coherent = abs(beta)^2 * L * v_norm_sq * gain_coherent / sigma2;

    PD_orth_vs_SNR(idx) = 1 - ncx2cdf(eta_fixed, 2, rho_orth);
    PD_coherent_vs_SNR(idx) = 1 - ncx2cdf(eta_fixed, 2, rho_coherent);
end

%% Plot style

%% Figure 1: P_D versus P_FA

%% ===========================================
width = 6;%设置图宽，这个不用改
height = 4;%设置图高，这个不用改
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
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

semilogx(PFA_vec, PD_orth_vs_PFA, 'b-', 'LineWidth', linewidth); hold on;
semilogx(PFA_vec, PD_coherent_vs_PFA, 'r--', 'LineWidth', linewidth);

h_legend = legend('Orthogonal signal', 'Coherent signal', 'Location', 'northeast', 'Interpreter', 'latex');

legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','best');
labelsize = 14;

xlabel('$P_{\mathrm{FA}}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('$P_{\mathrm{D}}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex'); 

xlim([min(PFA_vec), max(PFA_vec)]);
ylim([0, 1]);

%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');
set(get(gca, 'XAxis'), 'FontSize', 12);  % 调整坐标轴刻度标签（tick labels）的字体大小
set(get(gca, 'YAxis'), 'FontSize', 12);
%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.1, 0.1, 0.87, 0.86]); 
print(gcf, 'Fig_PD_vs_PFA.pdf', '-dpdf', '-vector');

%% Figure 2: P_D versus SNR

% ===========================================
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
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

plot(SNR_dB_vec, PD_orth_vs_SNR, 'b-', 'LineWidth', linewidth); hold on;
plot(SNR_dB_vec, PD_coherent_vs_SNR, 'r--', 'LineWidth', linewidth);
h_legend = legend('Orthogonal signal', 'Coherent signal', 'Location', 'northeast', 'Interpreter', 'latex');

legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','best');
labelsize = 14;

xlabel('SNR(dB)', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('$P_{\mathrm{D}}$', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
xlim([min(SNR_dB_vec), max(SNR_dB_vec)]);
ylim([0, 1]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');
set(get(gca, 'XAxis'), 'FontSize', 12);  % 调整坐标轴刻度标签（tick labels）的字体大小
set(get(gca, 'YAxis'), 'FontSize', 12);
%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.1, 0.1, 0.87, 0.86]);

print(gcf, 'Fig_PD_vs_SNR.pdf', '-dpdf', '-vector');