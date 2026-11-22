%% beam_squint_demo.m
% 远场均匀线阵的 beam squint：固定相移与理想真时延补偿对比。
% 本程序是原理演示，不是对某篇论文图片的逐点复现。
% 不需要通信工具箱；理想 TTD 也可由逐频率全数字权重实现。
clear;
clc;
close all;

%% 1. 参数与参考方向
c = 299792458;
fc = 30e9;
B = 6e9;
N = 64;
d = c / (2 * fc);
theta0Deg = 45;
freqScan = linspace(fc - B / 2, fc + B / 2, 301);
angleScanDeg = 20:0.02:70;
elementPosition = (0:N-1).' * d;
width = 8;
height = 4;
linewidth = 2;
colors = {'#F65314', '#00A1F1', '#8A2BE2'};
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

% a(f,theta) = exp(-j*2*pi*f*r*sin(theta)/c)。
% 使用 a/N 作为接收合并权重，使理想峰值功率统一为 1。
a0 = exp(-1j * 2 * pi * fc * elementPosition * sind(theta0Deg) / c);
wPhase = a0 / N;
gainPhase = zeros(numel(freqScan), numel(angleScanDeg));
gainTTD = zeros(size(gainPhase));

%% 2. 计算两种方案的频率—角度响应
for iFreq = 1:numel(freqScan)
    f = freqScan(iFreq);
    A = exp(-1j * 2 * pi * f * elementPosition * sind(angleScanDeg) / c);

    % 固定相移：所有频率共用中心频率处的同一组权重。
    gainPhase(iFreq, :) = abs(wPhase' * A).^2;

    % 理想真时延：补偿相位随实际射频频率 f 线性变化。
    aTarget = exp(-1j * 2 * pi * f * elementPosition * sind(theta0Deg) / c);
    wTTD = aTarget / N;
    gainTTD(iFreq, :) = abs(wTTD' * A).^2;
end

gainPhaseDB = 10 * log10(max(gainPhase, 1e-6));
gainTTDDB = 10 * log10(max(gainTTD, 1e-6));
thetaTheoryDeg = asind((fc ./ freqScan) * sind(theta0Deg));
[~, peakIndex] = max(gainPhase, [], 2);
thetaNumericalDeg = reshape(angleScanDeg(peakIndex), [], 1);
peakAngleErrorDeg = max(abs(thetaNumericalDeg - thetaTheoryDeg.'));

%% 3. 图1：三个频率处的固定相移波束
figure(1);
hold on;
selectedIndex = [1, (numel(freqScan) + 1) / 2, numel(freqScan)];
legendLabels = cell(1, numel(selectedIndex));
for k = 1:numel(selectedIndex)
    iFreq = selectedIndex(k);
    plot(angleScanDeg, gainPhaseDB(iFreq, :), 'Color', colors{k}, 'LineWidth', linewidth);
    legendLabels{k} = sprintf('$f=%.0f~\\mathrm{GHz}$', freqScan(iFreq) / 1e9);
end
xline(theta0Deg, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');
xlim([20, 70]);
ylim([-40, 1]);
xticks(20:10:70);
yticks(-40:10:0);
xlabel('$\theta~(\mathrm{deg})$', 'Interpreter', 'latex');
ylabel('Normalized power (dB)', 'Interpreter', 'latex');
legend(legendLabels, 'Interpreter', 'latex', 'FontSize', 13, 'Location', 'northwest');
formatFigure(gca, width, height, false);
saveFigure(gcf, 'beam_squint_fig1');

%% 4. 图2：固定相移的频率—角度响应
figure(2);
imagesc(angleScanDeg, freqScan / 1e9, gainPhaseDB);
axis xy;
hold on;
plot(thetaTheoryDeg, freqScan / 1e9, 'w--', 'LineWidth', 1.5);
plot([theta0Deg, theta0Deg], [freqScan(1), freqScan(end)] / 1e9, 'w:', 'LineWidth', 1.5);
caxis([-40, 0]);
colormap(parula);
cb = colorbar;
cb.Label.String = 'Normalized power (dB)';
cb.FontSize = 12;
xlabel('$\theta~(\mathrm{deg})$', 'Interpreter', 'latex');
ylabel('$f~(\mathrm{GHz})$', 'Interpreter', 'latex');
xticks(20:10:70);
yticks(27:1:33);
formatFigure(gca, width, height, true);
saveFigure(gcf, 'beam_squint_fig2');

%% 5. 图3：理想真时延的频率—角度响应
figure(3);
imagesc(angleScanDeg, freqScan / 1e9, gainTTDDB);
axis xy;
hold on;
plot([theta0Deg, theta0Deg], [freqScan(1), freqScan(end)] / 1e9, 'w--', 'LineWidth', 1.5);
caxis([-40, 0]);
colormap(parula);
cb = colorbar;
cb.Label.String = 'Normalized power (dB)';
cb.FontSize = 12;
xlabel('$\theta~(\mathrm{deg})$', 'Interpreter', 'latex');
ylabel('$f~(\mathrm{GHz})$', 'Interpreter', 'latex');
xticks(20:10:70);
yticks(27:1:33);
formatFigure(gca, width, height, true);
saveFigure(gcf, 'beam_squint_fig3');

%% 6. 图4：真实目标方向上的增益损失与阵元数的关系
arraySizes = [16, 64, 128];
targetGain = zeros(numel(arraySizes), numel(freqScan));
for iArray = 1:numel(arraySizes)
    currentN = arraySizes(iArray);
    currentPosition = (0:currentN-1).' * d;
    currentWeight = exp(-1j * 2 * pi * fc * currentPosition * sind(theta0Deg) / c) / currentN;
    for iFreq = 1:numel(freqScan)
        aTarget = exp(-1j * 2 * pi * freqScan(iFreq) * currentPosition * sind(theta0Deg) / c);
        targetGain(iArray, iFreq) = abs(currentWeight' * aTarget).^2;
    end
end

figure(4);
hold on;
legendLabels = cell(1, numel(arraySizes) + 1);
for iArray = 1:numel(arraySizes)
    plot(freqScan / 1e9, 10 * log10(max(targetGain(iArray, :), 1e-6)), 'Color', colors{iArray}, 'LineWidth', linewidth);
    legendLabels{iArray} = sprintf('Fixed phase, $N=%d$', arraySizes(iArray));
end
plot(freqScan / 1e9, zeros(size(freqScan)), 'k--', 'LineWidth', linewidth);
legendLabels{end} = 'Ideal TTD';
xlim([freqScan(1), freqScan(end)] / 1e9);
ylim([-50, 1]);
xticks(27:1:33);
yticks(-50:10:0);
xlabel('$f~(\mathrm{GHz})$', 'Interpreter', 'latex');
ylabel('Power at $\theta_0=45^{\circ}$ (dB)', 'Interpreter', 'latex');
legend(legendLabels, 'Interpreter', 'latex', 'FontSize', 13, 'Location', 'south');
formatFigure(gca, width, height, false);
saveFigure(gcf, 'beam_squint_fig4');

%% 7. 数值核对
fprintf('Theoretical and scanned peak-angle error: %.4f deg\n', peakAngleErrorDeg);
for k = 1:numel(selectedIndex)
    iFreq = selectedIndex(k);
    f = freqScan(iFreq);
    aTarget = exp(-1j * 2 * pi * f * elementPosition * sind(theta0Deg) / c);
    targetPower = abs(wPhase' * aTarget)^2;
    fprintf('f = %.1f GHz, peak angle = %.4f deg, target gain = %.4f dB\n', f / 1e9, thetaTheoryDeg(iFreq), 10 * log10(max(targetPower, eps)));
end

%% 绘图辅助函数
function formatFigure(ax, width, height, hasColorbar)
    fig = ancestor(ax, 'figure');
    set(fig, 'Units', 'inches', 'Position', [1, 1, width, height], 'Color', 'white', 'Renderer', 'painters');
    set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, width, height], 'PaperSize', [width, height]);
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 16, 'LineWidth', 1, 'Layer', 'bottom');
    ax.XLabel.FontSize = 16;
    ax.YLabel.FontSize = 16;
    if hasColorbar
        set(ax, 'Units', 'normalized', 'Position', [0.11, 0.14, 0.71, 0.82]);
    else
        grid(ax, 'on');
        set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.2, 'Units', 'normalized', 'Position', [0.11, 0.14, 0.87, 0.82]);
    end
    drawnow;
end

function saveFigure(fig, filename)
    print(fig, filename, '-dpng', '-r600');
    print(fig, filename, '-dpdf', '-painters');
end