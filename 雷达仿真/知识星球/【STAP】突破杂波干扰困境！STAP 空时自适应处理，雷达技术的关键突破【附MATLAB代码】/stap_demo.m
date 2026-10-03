%% STAP_merged.m
% 两份 STAP 示例合并：统一采用“阵元在外、脉冲在内”的堆叠方式。
% s(theta,nu) = a(theta) ⊗ b(nu)，nu = f_d*T_r。
% 所有图统一使用两个固定角度、固定多普勒的点干扰机。
% 两组图共用阵列、目标、杂波及干扰加噪声协方差。

clear;
clc;
close all;

chineseFont = 'Noto Sans CJK SC';
set(groot, 'DefaultAxesFontName', chineseFont);
set(groot, 'DefaultTextFontName', chineseFont);

%% 1. 统一的物理参数与扫描网格

N = 16;                         % 接收阵元数
M = 16;                         % CPI 内的脉冲数
NM = N * M;

lambda = 0.03;                  % 波长 / m
d = lambda / 2;                 % 阵元间距 / m
v = 60;                         % 平台速度 / (m/s)
Tr = 1e-4;                      % PRI / s

% u = (d/lambda)*sin(theta)，静止地物满足 nu_c = beta*u。
beta = 2 * v * Tr / d;

thetaTargetDeg = 40;            % 采用第一份附件中的目标角度 / deg
nuTarget = 0.32;                % 目标归一化多普勒 / cycles per PRI
jammerAngleDeg = [-35, 40];     % 两个干扰机的方位角 / deg
nuPointJammer = [0.1, 0.3];     % 与 jammerAngleDeg 一一对应

noisePower = 1;
CNRdB = 40;
JNRdB = 35;
SNRdB = 0;                    % 仅用于“含目标总回波谱”中的目标功率

totalClutterPower = noisePower * 10^(CNRdB / 10);
jammerPower = noisePower * 10^(JNRdB / 10);
targetPower = noisePower * 10^(SNRdB / 10);

numClutterPatches = 181;
clutterAngleDeg = linspace(-90, 90, numClutterPatches);
angleScanDeg = -90:1:90;
nuScan = -0.5:0.005:0.5;

%% 2. 目标空时导向矢量
aTarget = spatialSteeringVector(N, d, lambda, thetaTargetDeg);
bTarget = dopplerSteeringVector(M, nuTarget);
sTarget = kron(aTarget, bTarget);

% 第二组总回波谱中显式加入秩 1 目标协方差。
Rt = targetPower * (sTarget * sTarget');

%% 3. 白噪声协方差矩阵
Rn = noisePower * eye(NM);

%% 4. 杂波协方差矩阵
clutterSpatialFreq = (d / lambda) * sind(clutterAngleDeg);
clutterDoppler = beta * clutterSpatialFreq;
clutterPatchPower = totalClutterPower / numClutterPatches;
Vclutter = zeros(NM, numClutterPatches);

for k = 1:numClutterPatches
    aClutter = spatialSteeringVector(N, d, lambda, clutterAngleDeg(k));
    bClutter = dopplerSteeringVector(M, clutterDoppler(k));
    Vclutter(:, k) = kron(aClutter, bClutter);
end

Rc = clutterPatchPower * (Vclutter * Vclutter');
Rc = (Rc + Rc') / 2;


%% 5. 两个固定角度、固定多普勒的点干扰机

RjPoint = zeros(NM);

for q = 1:numel(jammerAngleDeg)
    aJammer = spatialSteeringVector(N, d, lambda, jammerAngleDeg(q));
    bPointJammer = dopplerSteeringVector(M, nuPointJammer(q));
    sPointJammer = kron(aJammer, bPointJammer);
    RjPoint = RjPoint + jammerPower * (sPointJammer * sPointJammer');
end

RjPoint = (RjPoint + RjPoint') / 2;

%% 6. 第一份代码：四种场景的 STAP 最优权重

Rcase = cell(4, 1);
Rcase{1} = Rn;
Rcase{2} = Rn + Rc;
Rcase{3} = Rn + RjPoint;
Rcase{4} = Rn + Rc + RjPoint;

caseName = {'仅白噪声', '白噪声 + 杂波', ...
    '白噪声 + 点干扰机', '白噪声 + 杂波 + 点干扰机'};

Wstap = cell(4, 1);
responseDB = cell(4, 1);

for iCase = 1:4
    x = Rcase{iCase} \ sTarget;
    Wstap{iCase} = x / (sTarget' * x);
    response = computeSpaceTimeResponse(Wstap{iCase}, N, M, d, lambda, angleScanDeg, nuScan);
    responseDB{iCase} = max(20 * log10(max(response, eps)), -80);
end

%% 图1：四种环境下的二维响应

figure(1);
set(gcf, 'Color', 'w', 'Position', [100, 100, 1150, 780]);
layout1 = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for iCase = 1:4
    ax = nexttile(layout1);
    imagesc(ax, angleScanDeg, nuScan, responseDB{iCase});
    axis(ax, 'xy');
    axis(ax, 'tight');
    caxis(ax, [-80, 0]);
    decorateAngleDopplerAxes(ax, chineseFont);
    title(ax, caseName{iCase}, 'FontSize', 13);
    colorbar(ax);
    hold(ax, 'on');

    plot(ax, thetaTargetDeg, nuTarget, 'ro', 'MarkerSize', 7, 'LineWidth', 1.5);

    if iCase == 2 || iCase == 4
        nuRidge = beta * (d / lambda) * sind(angleScanDeg);
        validIndex = abs(nuRidge) <= 0.5;
        plot(ax, angleScanDeg(validIndex), nuRidge(validIndex), 'w--', 'LineWidth', 1);
    end

    if iCase == 3 || iCase == 4
        for q = 1:numel(jammerAngleDeg)
            plot(ax, jammerAngleDeg(q), nuPointJammer(q), 'bs', 'MarkerSize', 7, 'LineWidth', 1.5);
        end
    end

    hold(ax, 'off');
end

title(layout1, '点干扰环境下的 STAP 角度-多普勒响应', 'FontSize', 15);
drawnow;

%% 图2：均匀空时匹配与 Chebyshev 空时加窗

wUniform = sTarget / (sTarget' * sTarget);
spaceWindow = chebwin(N, 30);
timeWindow = chebwin(M, 30);
spaceTimeWindow = kron(spaceWindow, timeWindow);
wChebRaw = sTarget .* spaceTimeWindow;
wCheb = wChebRaw / (sTarget' * wChebRaw);

uniformResponse = computeSpaceTimeResponse(wUniform, N, M, d, lambda, angleScanDeg, nuScan);
chebResponse = computeSpaceTimeResponse(wCheb, N, M, d, lambda, angleScanDeg, nuScan);
uniformDB = max(20 * log10(max(uniformResponse, eps)), -80);
chebDB = max(20 * log10(max(chebResponse, eps)), -80);

figure(2);
set(gcf, 'Color', 'w', 'Position', [140, 120, 1200, 580]);
layout2 = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
windowResponses = {uniformDB, chebDB};
windowTitles = {'均匀空时匹配', 'Chebyshev 空时加窗'};

for iWindow = 1:2
    ax = nexttile(layout2);
    imagesc(ax, angleScanDeg, nuScan, windowResponses{iWindow});
    axis(ax, 'xy');
    axis(ax, 'tight');
    caxis(ax, [-80, 0]);
    decorateAngleDopplerAxes(ax, chineseFont);
    title(ax, windowTitles{iWindow}, 'FontSize', 13);
    colorbar(ax);
end

title(layout2, '固定空时加窗的二维响应对比', 'FontSize', 15);
drawnow;

%% 5. 第一份代码：输出 SINR

RinPoint = Rcase{4};
wSTAP = Wstap{4};
sinrMatched = abs(wUniform' * sTarget)^2 / real(wUniform' * RinPoint * wUniform);
sinrSTAP = abs(wSTAP' * sTarget)^2 / real(wSTAP' * RinPoint * wSTAP);

fprintf('双点干扰机组合场景输出 SINR 对比\n');
fprintf('常规空时匹配权重： %.2f dB\n', 10 * log10(sinrMatched));
fprintf('STAP 最优权重：     %.2f dB\n', 10 * log10(sinrSTAP));
fprintf('STAP SINR 增益：    %.2f dB\n\n', 10 * log10(sinrSTAP / sinrMatched));

%% 6. 第二份代码：双点干扰场景中的总回波谱与 SINR 损失

Rpoint = Rcase{4};           % STAP 权重和 SINR 损失不包含目标协方差
Rtotal = Rpoint + Rt;         % DBF/Capon 总回波谱包含目标协方差

% 第二份原代码是 b ⊗ a、负指数。本脚本将目标、杂波、干扰和扫描
% 全部统一为 a ⊗ b、正指数；协方差和权重因此保持相同的排列约定。
% 对整个向量和矩阵使用相同的置换时，两种堆叠顺序给出相同的谱。
P_dbf = zeros(numel(nuScan), numel(angleScanDeg));
P_capon = zeros(size(P_dbf));
SINR_loss = zeros(size(P_dbf));
Ypoint = zeros(size(P_dbf));

xPoint = Rpoint \ sTarget;
wPoint = xPoint / (sTarget' * xPoint);

for iAngle = 1:numel(angleScanDeg)
    aScan = spatialSteeringVector(N, d, lambda, angleScanDeg(iAngle));

    for iNu = 1:numel(nuScan)
        bScan = dopplerSteeringVector(M, nuScan(iNu));
        scanVector = kron(aScan, bScan);

        % 常规 DBF：扫描方向上的总回波功率。
        P_dbf(iNu, iAngle) = real(scanVector' * Rtotal * scanVector);

        % Capon：用总回波协方差求空间-多普勒功率谱。
        P_capon(iNu, iAngle) = 1 / real(scanVector' * (Rtotal \ scanVector));

        % STAP 权重对当前扫描方向的响应。
        Ypoint(iNu, iAngle) = wPoint' * scanVector;

        % SINR 损失：有干扰时的最优 SINR 与仅有白噪声时的最优 SNR 之比。
        SINR_current = real(scanVector' * (Rpoint \ scanVector));
        SNRopt = real(scanVector' * scanVector) / noisePower;
        SINR_loss(iNu, iAngle) = SINR_current / SNRopt;
    end
end

dbfDB = 10 * log10(max(P_dbf, eps));
caponDB = 10 * log10(max(P_capon, eps));
pointResponseDB = 10 * log10(max(abs(Ypoint).^2, eps));
lossDB = 10 * log10(max(SINR_loss, eps));

%% 图3：干扰加噪声协方差矩阵特征值谱

singularValues = svd(Rpoint);

figure(3);
set(gcf, 'Color', 'w');
plot(1:NM, 10 * log10(singularValues), 'LineWidth', 1.3);
grid on;
xlabel('特征值序号');
ylabel('特征值 / dB');
title('双点干扰场景的干扰加噪声协方差矩阵特征值谱');
drawnow;

%% 图4：处理前总回波 DBF 二维功率谱

figure(4);
set(gcf, 'Color', 'w');
imagesc(angleScanDeg, nuScan, dbfDB);
axis xy;
axis tight;
decorateAngleDopplerAxes(gca, chineseFont);
colorbar;
title('STAP 处理前总回波的 DBF 二维功率谱');
drawnow;

%% 图5：处理前总回波 DBF 三维功率谱

figure(5);
set(gcf, 'Color', 'w');
surf(angleScanDeg, nuScan, dbfDB, 'EdgeColor', 'none');
view(-37.5, 30);
axis tight;
decorateAngleDopplerAxes(gca, chineseFont, true);
zlabel('功率 / dB');
colorbar;
title('STAP 处理前总回波的 DBF 三维功率谱');
drawnow;

%% 图6：处理前总回波 Capon 二维功率谱

figure(6);
set(gcf, 'Color', 'w');
imagesc(angleScanDeg, nuScan, caponDB);
axis xy;
axis tight;
decorateAngleDopplerAxes(gca, chineseFont);
colorbar;
title('STAP 处理前总回波的 Capon 二维功率谱'); hold on;
plot(jammerAngleDeg, nuPointJammer, 'ks', 'MarkerSize', 11, 'LineWidth', 2);
hold off;
drawnow;

%% 图7：处理前总回波 Capon 三维功率谱

figure(7);
set(gcf, 'Color', 'w');
surf(angleScanDeg, nuScan, caponDB, 'EdgeColor', 'none');
view(-37.5, 30);
axis tight;
decorateAngleDopplerAxes(gca, chineseFont, true);
zlabel('功率 / dB');
colorbar;
title('STAP 处理前总回波的 Capon 三维功率谱');
drawnow;

%% 图8：双点干扰环境下的最优 STAP 二维响应

figure(8);
set(gcf, 'Color', 'w');
imagesc(angleScanDeg, nuScan, pointResponseDB);
axis xy;
axis tight;
decorateAngleDopplerAxes(gca, chineseFont);
colorbar;
title('双点干扰环境下的 STAP 最优权重二维响应');
drawnow;

%% 图9：双点干扰环境下的最优 STAP 三维响应

figure(9);
set(gcf, 'Color', 'w');
mesh(angleScanDeg, nuScan, pointResponseDB);
view(-37.5, 30);
axis tight;
decorateAngleDopplerAxes(gca, chineseFont, true);
zlabel('响应功率 / dB');
colorbar;
title('双点干扰环境下的 STAP 最优权重三维响应');
drawnow;

%% 图10：零角度处的 SINR 损失曲线

zeroAngleIndex = find(angleScanDeg == 0, 1);

figure(10);
set(gcf, 'Color', 'w');
plot(nuScan, lossDB(:, zeroAngleIndex), 'LineWidth', 1.3);
grid on;
xlabel('$\nu=f_{\mathrm{d}}T_{\mathrm{r}}$', 'Interpreter', 'latex');
ylabel('SINR 损失 / dB');
title('零角度处的 SINR 损失');
drawnow;

%% 图11：整个角度-多普勒平面的 SINR 损失

figure(11);
set(gcf, 'Color', 'w');
mesh(angleScanDeg, nuScan, lossDB);
view(-37.5, 30);
axis tight;
decorateAngleDopplerAxes(gca, chineseFont, true);
zlabel('SINR 损失 / dB');
colorbar;
title('双点干扰环境下的 SINR 损失');
drawnow;

%% 局部函数

function a = spatialSteeringVector(N, d, lambda, thetaDeg)
    elementIndex = (0:N-1).';
    u = (d / lambda) * sind(thetaDeg);
    a = exp(1j * 2 * pi * elementIndex * u);
end

function b = dopplerSteeringVector(M, nu)
    pulseIndex = (0:M-1).';
    b = exp(1j * 2 * pi * pulseIndex * nu);
end

function response = computeSpaceTimeResponse(w, N, M, d, lambda, angleScanDeg, nuScan)
    response = zeros(numel(nuScan), numel(angleScanDeg));
    pulseIndex = (0:M-1).';
    Bscan = exp(1j * 2 * pi * pulseIndex * nuScan);

    for iAngle = 1:numel(angleScanDeg)
        aScan = spatialSteeringVector(N, d, lambda, angleScanDeg(iAngle));
        Sscan = kron(aScan, Bscan);
        response(:, iAngle) = abs(w' * Sscan).';
    end
end

function decorateAngleDopplerAxes(ax, chineseFont, is3D)
    if nargin < 3
        is3D = false;
    end

    ax.FontName = chineseFont;
    ax.FontSize = 11;
    xlabel(ax, '角度 \theta / deg', 'FontName', chineseFont, 'Interpreter', 'tex', 'FontSize', 13);
    yLabel = ylabel(ax, '$\nu=f_{\mathrm{d}}T_{\mathrm{r}}$', 'Interpreter', 'latex', 'FontSize', 13);

    % 所有三维图均使用 view(-37.5,30)，旋转标签以贴近屏幕上的 y 轴。
    if is3D
        yLabel.Rotation = -32;
        yLabel.Units = 'data';
        yLabel.HorizontalAlignment = 'center';
        yLabel.VerticalAlignment = 'middle';
    
        xLimits = xlim(ax);
        yLimits = ylim(ax);
        zLimits = zlim(ax);
        yLabel.Position = [xLimits(1) - 0.12 * diff(xLimits), mean(yLimits), zLimits(1)];
    end
end