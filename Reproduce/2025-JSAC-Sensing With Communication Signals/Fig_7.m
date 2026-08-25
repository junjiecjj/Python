

%% Fig. 7 in "Sensing With Communication Signals: From Information Theory to Signal Processing"
%
% Equations used:
%   Eq. (34): pulse-shaped continuous-time transmit signal (discretized here)
%   Eq. (35): x = U*s, with U = F_N^H for OFDM
%   Eq. (40b): three-target sensing echo
%   Eq. (47): matched filtering using the complete transmitted signal

clear;
clc;
close all;

rng(42);

%% Parameters explicitly specified around Fig. 7
Order = 16;                           % 16-PSK
SNRdB = 20;                           % input echo SNR
targetRange = [10, 20, 25];           % [m]
Q = length(targetRange);
Iter = 1000;                          % number of Monte Carlo trials

%% Parameters not explicitly reported for Fig. 7 in the paper
% These parameters are collected here instead of being presented as paper values.
N = 1024;                              % number of OFDM symbols/subcarriers
L = 20;                               % oversampling factor
alpha = 0.3;                         % RRC roll-off factor
span = 20;                            % RRC span in symbols
rangeSampleSpacing = 0.02;           % range represented by one sample [m]

% The peak levels are read approximately from the published Fig. 7.
targetAmplitude_dB = [0, -10, -30];
targetPhase = [0, 0, 0];
gamma = 10.^(targetAmplitude_dB/20).*exp(1j*targetPhase);

%% Fixed matrices and pulse
FFTmatrix = exp(-1j*2*pi*(0:N-1).'*(0:N-1)/N)/sqrt(N);
U = FFTmatrix';
p = rcosdesign(alpha, span, L, 'sqrt').';
p = p/norm(p);

targetDelaySample = round(targetRange/rangeSampleSpacing);
maxTargetDelaySample = max(targetDelaySample);

lengthXTilde = L*N + length(p) - 1;
lengthYs = lengthXTilde + maxTargetDelaySample;
lengthYMatched = lengthYs + lengthXTilde - 1;

% For conv(y, conj(flip(x))), the zero-delay peak occurs at length(xTilde).
delaySample = (1:lengthYMatched).' - lengthXTilde;
rangeAxis = delaySample*rangeSampleSpacing;
plotIndex = rangeAxis >= 0 & rangeAxis <= 35;

%% Monte Carlo simulation
% Preserve the complete arrays from every trial before ensemble averaging.
SimRangeProfile = zeros(Iter, lengthYMatched);
SimTargetProfile = zeros(Iter, lengthYMatched, Q);

for ii = 1:Iter
    %% Eq. (35): OFDM time-domain samples x = U*s = F_N^H*s
    symbolIndex = randi([0, Order - 1], N, 1);
    % s = exp(1j*2*pi*symbolIndex/Order);
    % s = qammod(symbolIndex, Order, 'gray', 'UnitAveragePower', true);
    s = pskmod(symbolIndex, Order, pi/Order);
    x = U*s;

    %% Eq. (34): RRC pulse shaping
    xup = complex(zeros(L*N, 1));
    xup(1:L:end) = x;
    xTilde = conv(xup, p, 'full');

    %% Eq. (40b): sensing echo from three delayed targets
    ysTarget = complex(zeros(lengthYs, Q));

    for q = 1:Q
        startIndex = targetDelaySample(q) + 1;
        endIndex = targetDelaySample(q) + length(xTilde);
        ysTarget(startIndex:endIndex,q) = gamma(q)*xTilde;
    end

    ysNoiseless = sum(ysTarget, 2);

    signalPower = mean(abs(ysNoiseless).^2);
    noisePower = signalPower/10^(SNRdB/10);
    zs = sqrt(noisePower/2)*(randn(size(ysNoiseless)) + 1j*randn(size(ysNoiseless)));
    ys = ysNoiseless + zs;

    %% Eq. (47): matched filter uses the complete transmitted signal
    matchedFilter = conj(flipud(xTilde));

    for q = 1:Q
        yMatchedTarget = conv(ysTarget(:,q), matchedFilter, 'full');
        SimTargetProfile(ii,:,q) = abs(yMatchedTarget.').^2;
    end

    yMatched = conv(ys, matchedFilter, 'full');
    SimRangeProfile(ii,:) = abs(yMatched.').^2;
end

%% Ensemble average of the squared matched-filter magnitude
AveRangeProfile = mean(SimRangeProfile, 1);
AveTargetProfile = squeeze(mean(SimTargetProfile, 1));

normalization = max(AveRangeProfile);
rangeProfile_dB = 10*log10(AveRangeProfile/normalization + eps);
targetProfile_dB = 10*log10(AveTargetProfile/normalization + eps);


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

plot(rangeAxis(plotIndex), targetProfile_dB(plotIndex,1), ':',  'Color','#F65314', 'LineWidth', 1.5); hold on;

plot(rangeAxis(plotIndex), targetProfile_dB(plotIndex,2), ':', 'Color', '#00A1F1', 'LineWidth', 1.5);

plot(rangeAxis(plotIndex), targetProfile_dB(plotIndex,3), ':', 'Color','#8A2BE2', 'LineWidth', 1.5);

plot(rangeAxis(plotIndex), rangeProfile_dB(plotIndex), '-', 'Color', '#A9A9A9', 'LineWidth', 1.5);

% 设置坐标轴的数字大小，包括xlabel/ylabel文字(坐标轴标注)大小.同时影响图例、标题等,除非它们被单独设置。
% 所以一开始就使用这行先设置刻度字体字号，然后在后面在单独设置坐标轴标注、图例、标题等的 字体字号。
set(gca, 'FontSize',16,'FontName','Times New Roman');
h_legend =  legend('Target 1', 'Target 2', 'Target 3', 'Range Profile', 'Interpreter', 'latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
% set(h_legend,'Interpreter','latex') %  'box','off');
% h_legend.Interpreter = 'latex';
labelsize = 16;
xlabel('Range [m]', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');
ylabel('Amplitude [dB]', 'FontSize', labelsize, 'FontName', 'Times New Roman', 'Interpreter', 'latex');

xlim([0, 35]);
ylim([-50, 0]);
xticks(0:5:35);
yticks(-80:10:0);
%----- Grid 设置----------------
grid on;
set(gca,'GridLineStyle', '--', 'Gridalpha',0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer','bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.12, 0.87, 0.86]);

% print(gcf, 'Fig7_JSAC.pdf', '-dpdf', '-vector');
