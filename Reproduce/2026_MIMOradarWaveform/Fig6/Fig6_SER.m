


clc;
clear;
close all;
addpath('./functions_2018ICC');
addpath('./functions_2007TSP_OnProb');
addpath('./functions_2008TAES_CrossCorre');
addpath('./functions_2008TSP_WaveformSynthesis');


%% 1. 参数设置（示例，可修改）
Kc = 4;                      % # of users
M = 16;                     % 天线数
L = 100;                     % # of Communication Frame
Pt  = 1;
c = ones(M, 1) * Pt/M;        % 对角元固定值
% c = rand(M, 1)

d = 0.5;
lambda = 2 * d;
pos = (0:M-1) * d;
normalizedPos = pos / lambda;

afun = @(theta) exp(1j * pi * (0:M-1)' * sind(theta));  % M×1

%% Desired Beampattern
theta_est = [-60, 0, 60];   % 目标角度估计（度）
Kt = length(theta_est);      % 目标个数

Delta = 5;
theta_grid = -90:0.1:90;
P_des = zeros(size(theta_grid));
% Desired beam pattern
idx = false(size(theta_grid));
for i = 1:numel(theta_est)
    idx = idx | theta_grid >= theta_est(i)-Delta & theta_grid <= theta_est(i)+Delta;
end
P_des(idx) = 1;

%% Omni-Directional Beampattern
OmniRd = (Pt / M) * eye(M);

%% Directional Beampattern
%  文献1：On Probing Signal Design For MIMO Radar, C. Beampattern Matching Design
%  diag(R)=1/M, trace(R)=1, wc=0
w_l = ones(length(theta_grid), 1);
w_c = 0;
[DirectRd1, alpha1, ~] = BeampatternMatchingDesign(c, M, w_l, w_c, theta_est, theta_grid, P_des);
P_des1 = P_des * alpha1;
fprintf('trace(DirectRd1) = %.6f\n',  trace(DirectRd1));
%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  helperMMSECovariance 默认 diag(R)=1, trace(R)=M,为了和文献1对齐，将 R 除以 M，使 trace(R)=1
DirectRd2 = helperMMSECovariance(normalizedPos, P_des, theta_grid);
DirectRd2 = DirectRd2 / M;
DirectRd2 = projectToPSD(DirectRd2);
DirectRd2 = DirectRd2 + 1e-10 * eye(size(M));
%  文献2：Transmit Beamforming for MIMO Radar Systems using Signal Cross-Correlation, A. Squared Error Optimization
%  不用 cos(theta) 权重，不做积分归一化，不用 barrier/Newton，直接 CVX 最小化二范数
[DirectRd3, b] = helperMMSECovariance_direct(normalizedPos, P_des, theta_grid, Pt); 
fprintf('trace(DirectRd3) = %.6f\n',  trace(DirectRd3));

SNRdB = -5:1:18;
N0 = Pt ./ 10.^(SNRdB/10);


%% Choose Directional Covariance Matrix
DirectRd = DirectRd1;
rho = 0.2;   % Tradeoff Settings
par = 1.1;                          % Parameter that controls low PAR

%% Monte Carlo Simulation

%% Initialization
Q = 4;
Iters = 5000;

OmniStrictSERArray = zeros(Iters, length(SNRdB));
OmniTradeoffSERTolArray = zeros(Iters, length(SNRdB));
OmniTradeoffSERPerAntArray = zeros(Iters, length(SNRdB));
DirectStrictSERArray = zeros(Iters, length(SNRdB));
DirectTradeoffSERTolArray = zeros(Iters, length(SNRdB));
DirectTradeoffSERPerAntArray = zeros(Iters, length(SNRdB));
ZeroMUISERArray = zeros(Iters, length(SNRdB));

OmniStrictBPArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPTolArray = zeros(Iters, length(theta_grid));
OmniTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));
DirectStrictBPArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPTolArray = zeros(Iters, length(theta_grid));
DirectTradeoffBPPerAntArray = zeros(Iters, length(theta_grid));

%% Monte Carlo Simulation
for iter = 1:Iters
    fprintf('Monte Carlo iteration: %d / %d\n', iter, Iters);
    for idxSNR = 1:length(SNRdB)
        H = (randn(Kc, M) + 1j * randn(Kc, M)) / sqrt(2);

        data = randi([0, Q - 1], Kc, L);
        S = pskmod(data, Q, pi / Q, 'gray');

        OmniStrictX = strict_waveform(H, S, OmniRd, L);
        DirectStrictX = strict_waveform(H, S, DirectRd, L);

        % 如果你想用波形合成生成严格雷达波形，可以替换为下面两行
        % OmniStrictX = WaveformSynthesisXoptimR(OmniRd, L, par);
        % DirectStrictX = WaveformSynthesisXoptimR(DirectRd, L, par);

        OmniTradeoffTolX = algorithm1_tradeoff(H, S, OmniStrictX, Pt, rho);
        DirectTradeoffTolX = algorithm1_tradeoff(H, S, DirectStrictX, Pt, rho);

        OmniTradeoffPerAntX = RiemannianGradientDescent(H, S, OmniStrictX, Pt, rho);
        DirectTradeoffPerAntX = RiemannianGradientDescent(H, S, DirectStrictX, Pt, rho);

        OmniStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniStrictX, data, Q, N0(idxSNR));
        OmniTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffTolX, data, Q, N0(idxSNR));
        OmniTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, OmniTradeoffPerAntX, data, Q, N0(idxSNR));
        DirectStrictSERArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectStrictX, data, Q, N0(idxSNR));
        DirectTradeoffSERTolArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffTolX, data, Q, N0(idxSNR));
        DirectTradeoffSERPerAntArray(iter, idxSNR) = qpsk_ser_from_waveform(H, DirectTradeoffPerAntX, data, Q, N0(idxSNR));
        ZeroMUISERArray(iter, idxSNR) = qpsk_ser_zero_mui(S, data, Q, N0(idxSNR));
    end
end

%% Average Results
OmniStrictSER = mean(OmniStrictSERArray, 1);
OmniTradeoffSERTol = mean(OmniTradeoffSERTolArray, 1);
OmniTradeoffSERPerAnt = mean(OmniTradeoffSERPerAntArray, 1);
DirectStrictSER = mean(DirectStrictSERArray, 1);
DirectTradeoffSERTol = mean(DirectTradeoffSERTolArray, 1);
DirectTradeoffSERPerAnt = mean(DirectTradeoffSERPerAntArray, 1);
ZeroMUISER = mean(ZeroMUISERArray, 1);

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
% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',fontsize,'FontName','Times New Roman');

semilogy(SNRdB, OmniStrictSER, 'b-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, OmniTradeoffSERTol, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, OmniTradeoffSERPerAnt, 'b--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectStrictSER, 'r-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERTol, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, DirectTradeoffSERPerAnt, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
semilogy(SNRdB, ZeroMUISER, 'k--v', 'LineWidth', 1.5, 'MarkerSize', 7);

h_legend =  legend('Omni-Strict', ...
       'Omni-Tradeoff-Tol, $\rho$ = 0.2', ...
       'Omni-Tradeoff-Per, $\rho$ = 0.2', ...
       'Directional-Strict', ...
       'Directional-Tradeoff-Tol, $\rho$ = 0.2', ...
       'Directional-Tradeoff-Per, $\rho$ = 0.2', ...
       'Zero MUI', ...
       'Location', 'SouthWest',...
       'Interpreter', 'latex');

legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','best');
labelsize = 14;

xlabel('Transmit SNR (dB)', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel("SER", 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
xlim([min(SNRdB), max(SNRdB)]);
ylim([10^(-5), 10^(0)]);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.1, 0.1, 0.87, 0.86]);
print(gcf, 'Fig_6_ser.pdf', '-dpdf', '-vector');




%% Figure: SER
% figure(1);
% semilogy(SNRdB, OmniStrictSER, 'b-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, OmniTradeoffSERTol, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, OmniTradeoffSERPerAnt, 'b--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, DirectStrictSER, 'r-x', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, DirectTradeoffSERTol, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, DirectTradeoffSERPerAnt, 'r--d', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
% semilogy(SNRdB, ZeroMUISER, 'k--v', 'LineWidth', 1.5, 'MarkerSize', 7);
% grid on;
% xlabel('Transmit SNR (dB)');
% ylabel('SER');
% legend('Omni-Strict', ...
%        'Omni-Tradeoff-Total, \rho = 0.2', ...
%        'Omni-Tradeoff-perAnt, \rho = 0.2', ...
%        'Directional-Strict', ...
%        'Directional-Tradeoff-Total, \rho = 0.2', ...
%        'Directional-Tradeoff-perAnt, \rho = 0.2', ...
%        'Zero MUI', ...
%        'Location', 'SouthWest');
% xlim([min(SNRdB), max(SNRdB)]);
% ylim([1e-5, 1]);







