

%% 基于MATLAB2024a的双偏振相干光通信64QAM概率整形系统仿真主脚本

clear all;
clc;
close all;
addpath(genpath(pwd));

rng(2026);

%% ========================= 参数设置 =========================
cfg.M              = 64;             % 64QAM
cfg.bitsPerSym     = log2(cfg.M);    % 每个符号比特数
cfg.nu             = 0.060;          % Maxwell-Boltzmann整形参数，越大整形越强
cfg.snrScatterDb   = 18;             % 星座展示用SNR
cfg.SNRdB_GMI      = -20:1:30;
cfg.SNRdB_BER      = 0:2:22;         % BER曲线
cfg.SNRdB_Optical  = 6:2:24;
cfg.Nscatter       = 25000;          % 星座图用符号数
cfg.Nair           = 12000;          % AIR/GMI/MI仿真符号数
cfg.Nber           = 30000;          % BER仿真符号数
cfg.nuSweep        = 0:0.01:0.12;    % 整形参数扫描
cfg.nuOpticalList  = [0.00 0.03 0.06 0.09];
cfg.symbolRate     = 32e9;           % 32 Gbaud，双偏振相干光场景
cfg.numPolar       = 2;              % 双偏振
cfg.saveFigures    = true;           % 是否自动保存图像
cfg.outDir         = 'figures_out';  % 图像输出目录

if cfg.saveFigures && ~exist(cfg.outDir,'dir')
    mkdir(cfg.outDir);
end

%% ========================= 构造64QAM星座 =========================
[constRaw, bitLabels, axisLevels] = ps64qam_build_constellation();

% 均匀分布与概率整形分布
pmfUniform = ones(cfg.M,1) / cfg.M;
pmfShaped  = ps64qam_mb_pmf(constRaw, cfg.nu);

% 为公平比较，分别把“平均发射功率”归一化到1
[constUniform, scaleUniform] = ps64qam_normalize_avg_power(constRaw, pmfUniform);
[constShaped,  scaleShaped ] = ps64qam_normalize_avg_power(constRaw, pmfShaped);

fprintf('================ 参数与统计信息 ================\n');
fprintf('整形参数 nu                = %.4f\n', cfg.nu);
fprintf('均匀64QAM归一化缩放因子   = %.6f\n', scaleUniform);
fprintf('整形64QAM归一化缩放因子   = %.6f\n', scaleShaped);
fprintf('整形后输入熵 H(X)          = %.4f bit/符号\n', ps64qam_entropy(pmfShaped));
fprintf('================================================\n\n');

%% ========================= 1：概率分布三维图 =========================
probGrid = ps64qam_probability_grid(constRaw, pmfShaped, axisLevels);

figure(1);
b = bar3(axisLevels, probGrid, 0.85);
for kk = 1:length(b)
    zdata = b(kk).ZData;
    b(kk).CData = zdata;
    b(kk).FaceColor = 'interp';
end
title('64QAM概率整形后的二维概率分布（三维柱状图）');
xlabel('I轴幅度电平');
ylabel('Q轴幅度电平');
zlabel('符号概率');
xticks(axisLevels);
yticks(axisLevels);
view(-38,28);
grid on;
colormap(turbo);
cb = colorbar;
cb.Label.String = '概率大小';
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig1_三维概率分布图.png'));
end

%% ========================= 2：整形星座图 =========================
[txIdxScatter, txSymScatter] = ps64qam_draw_symbols(constShaped, pmfShaped, cfg.Nscatter);
rxScatter = ps64qam_awgn(txSymScatter, cfg.snrScatterDb);

figure(2);
scatter(real(rxScatter), imag(rxScatter), 5, [0.82 0.82 0.82], 'r*');
hold on;
markerSize = 30 + 300 * pmfShaped / max(pmfShaped);
% scatter(real(constShaped), imag(constShaped), markerSize, 'r', 'filled');
title(sprintf('64QAM概率整形星座图（SNR = %d dB）', cfg.snrScatterDb));
xlabel('同相分量 I');
ylabel('正交分量 Q');
axis equal;
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig2_整形星座图.png'));
end

%% ========================= 3：归一化GMI曲线 =========================
[miShaped, gmiShaped, HxShaped] = ps64qam_air_curve(constShaped, bitLabels, pmfShaped, cfg.SNRdB_GMI, cfg.Nair);
gmiNormToHx = max(min(gmiShaped / HxShaped, 1), 0);

figure(3);
plot(cfg.SNRdB_GMI, gmiNormToHx, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
title('归一化GMI');
xlabel('SNR (dB)');
ylabel('归一化GMI');
ylim([0 1.05]);
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig3_归一化GMI曲线.png'));
end

%% ========================= 4：均匀64QAM星座与整形64QAM星座对比 =========================
figure(4);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
scatter(real(constUniform), imag(constUniform), 65, 'b', 'filled');
title('均匀64QAM理想星座');
xlabel('I');
ylabel('Q');
axis equal;
grid on;

nexttile;
scatter(real(constShaped), imag(constShaped), markerSize, 'r', 'filled');
title('概率整形64QAM理想星座');
xlabel('I');
ylabel('Q');
axis equal;
grid on;

set(findall(gcf,'Type','axes'),'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig4_均匀与整形星座对比.png'));
end

%% ========================= 5：I轴一维概率分布对比 =========================
[pAxisUniform, pAxisShaped] = ps64qam_axis_pmf(axisLevels, constRaw, pmfUniform, pmfShaped);

figure(5);
stem(axisLevels, pAxisUniform, 'filled', 'LineWidth', 1.4);
hold on;
stem(axisLevels + 0.12, pAxisShaped, 'filled', 'LineWidth', 1.4);
title('I轴一维幅度分布：均匀分布 vs Maxwell-Boltzmann分布');
xlabel('I轴幅度电平');
ylabel('概率');
legend('均匀64QAM','概率整形64QAM','Location','best');
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig5_I轴概率分布对比.png'));
end

%% ========================= 6：二维概率热力图 =========================
figure(6);
imagesc(axisLevels, axisLevels, probGrid);
axis xy;
axis equal tight;
xticks(axisLevels);
yticks(axisLevels);
title('64QAM整形后的二维概率热力图');
xlabel('I轴幅度电平');
ylabel('Q轴幅度电平');
cb = colorbar;
cb.Label.String = '概率大小';
colormap(parula);
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig6_二维概率热力图.png'));
end

%% ========================= 7：MI与GMI对比 =========================
[miUniform, gmiUniform, HxUniform] = ps64qam_air_curve(constUniform, bitLabels, pmfUniform, cfg.SNRdB_GMI, cfg.Nair);

figure(7);
plot(cfg.SNRdB_GMI, miUniform,  '-', 'LineWidth', 1.6); hold on;
plot(cfg.SNRdB_GMI, gmiUniform, '--', 'LineWidth', 1.6);
plot(cfg.SNRdB_GMI, miShaped,   '-', 'LineWidth', 1.6);
plot(cfg.SNRdB_GMI, gmiShaped,  '--', 'LineWidth', 1.6);
title('均匀64QAM与概率整形64QAM的MI/GMI对比');
xlabel('SNR (dB)');
ylabel('信息率 (bit/符号)');
legend('均匀64QAM-MI','均匀64QAM-GMI','整形64QAM-MI','整形64QAM-GMI','Location','southeast');
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig7_MI与GMI对比.png'));
end

%% ========================= 8：BER性能对比 =========================
berUniform = ps64qam_ber_curve(constUniform, bitLabels, pmfUniform, cfg.SNRdB_BER, cfg.Nber);
berShaped  = ps64qam_ber_curve(constShaped,  bitLabels, pmfShaped,  cfg.SNRdB_BER, cfg.Nber);

figure(8);
semilogy(cfg.SNRdB_BER, berUniform, 'o-', 'LineWidth', 1.6, 'MarkerSize', 6); hold on;
semilogy(cfg.SNRdB_BER, berShaped,  's-', 'LineWidth', 1.6, 'MarkerSize', 6);
title('BER性能对比：均匀64QAM vs 概率整形64QAM');
xlabel('SNR (dB)');
ylabel('BER');
legend('均匀64QAM（ML/MAP统一框架）','概率整形64QAM（带先验MAP）','Location','southwest');
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig8_BER性能对比.png'));
end

%% ========================= 9：整形参数扫描 =========================
entropyList = zeros(length(cfg.nuSweep),1);
avgPowerList = zeros(length(cfg.nuSweep),1);

for k = 1:length(cfg.nuSweep)
    pmfTemp = ps64qam_mb_pmf(constRaw, cfg.nuSweep(k));
    entropyList(k) = ps64qam_entropy(pmfTemp);
    avgPowerList(k) = sum(pmfTemp .* abs(constRaw).^2);
end

figure(9);
yyaxis left;
plot(cfg.nuSweep, entropyList, 'o-', 'LineWidth', 1.6, 'MarkerSize', 5);
ylabel('输入熵 H(X) (bit/符号)');

yyaxis right;
plot(cfg.nuSweep, avgPowerList, 's-', 'LineWidth', 1.6, 'MarkerSize', 5);
ylabel('未归一化平均符号能量');

title('整形参数变化对输入熵和平均能量的影响');
xlabel('Maxwell-Boltzmann整形参数 \nu');
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig9_整形参数扫描.png'));
end

%% ========================= 10：光通信等效场景——不同整形强度的归一化GMI =========================
figure(10);
hold on;

legendText = cell(length(cfg.nuOpticalList),1);

for k = 1:length(cfg.nuOpticalList)
    nuNow = cfg.nuOpticalList(k);
    if nuNow == 0
        pmfNow = pmfUniform;
        constNow = constUniform;
    else
        pmfNow = ps64qam_mb_pmf(constRaw, nuNow);
        [constNow, ~] = ps64qam_normalize_avg_power(constRaw, pmfNow);
    end

    [~, gmiNow, HxNow] = ps64qam_air_curve(constNow, bitLabels, pmfNow, cfg.SNRdB_Optical, round(cfg.Nair*0.8));
    gmiNormNow = max(min(gmiNow / HxNow, 1), 0);
    plot(cfg.SNRdB_Optical, gmiNormNow, 'o-', 'LineWidth', 1.5, 'MarkerSize', 5);

    if nuNow == 0
        legendText{k} = '均匀64QAM';
    else
        legendText{k} = sprintf('概率整形64QAM, \\nu = %.2f', nuNow);
    end
end

title('光通信等效AWGN场景：不同整形强度的归一化GMI');
xlabel('等效SNR / OSNR映射后的SNR (dB)');
ylabel('归一化GMI');
ylim([0 1.05]);
legend(legendText,'Location','southeast');
grid on;
set(gca,'FontName','Microsoft YaHei','FontSize',11);

if cfg.saveFigures
    saveas(gcf, fullfile(cfg.outDir,'Fig10_光通信等效场景归一化GMI.png'));
end