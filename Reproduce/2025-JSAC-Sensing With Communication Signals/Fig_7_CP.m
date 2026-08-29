

%% Fig. 7 reproduction with CP added before pulse shaping
% Sensing With Communication Signals: From Information Theory to Signal Processing
% Transmitter: s -> x = U*s -> add CP -> upsample -> RRC pulse shaping
% Sensing channel: physical linear delays of the unique transmitted waveform
% Receiver: select the useful block -> periodic waveform matched filtering

clear;
clc;
close all;
rng(42);

%% Parameters used for the three-target example
Order = 16;                         % 16-PSK星座阶数
SNRdB = -10;                          % 去CP后的总无噪声回波与输入噪声之比，单位为dB
targetRange = [10,20,25];           % 三个目标的物理距离，单位为m
targetAmplitude_dB = [0,-10,-30];   % 三个目标复反射系数的幅度，单位为dB
targetPhase = [0,0,0];              % 三个目标复反射系数的相位，单位为rad
Q = length(targetRange);            % 目标数量
Iter = 1000;                        % 蒙特卡洛实验次数

%% Discrete waveform parameters
N = 4096;                           % 一个OFDM块中的子载波数，也等于IFFT输出样本数
L = 20;                             % pulse shaping的过采样倍数，每个符号对应L个高采样率样本
alpha = 0.35;                       % RRC脉冲的滚降系数
span = 20;                          % RRC脉冲的截断长度，单位为符号周期
rangeSampleSpacing = 0.025;         % 高采样率下一个采样点对应的单站雷达距离，单位为m
desiredCPRange = 40;                % 希望CP至少覆盖的物理距离，单位为m
c = 299792458;                      % 真空光速，单位为m/s

%% Target amplitudes
gamma = 10.^(targetAmplitude_dB/20).*exp(1j*targetPhase);   % dB幅度使用20而不是10转换为复幅度

%% Physical range, propagation delay and sample-index conversion
Fs = c/(2*rangeSampleSpacing);                       % 由DeltaR=c/(2Fs)反推出高采样率Fs
Ts = 1/Fs;                                           % 高采样率采样间隔，单位为s
symbolPeriod = L*Ts;                                 % 原始离散序列x相邻样本之间的时间间隔T=L*Ts
symbolRangeSpacing = L*rangeSampleSpacing;           % x中一个符号率样本对应的距离，单位为m
targetPropagationDelay = 2*targetRange/c;            % 单站雷达往返传播时延tau=2R/c
targetDelaySample = round(targetPropagationDelay/Ts);% 将物理时延量化为高采样率整数索引
targetRangeGrid = targetDelaySample*c*Ts/2;          % 整数时延索引实际对应的栅格距离
targetRangeError = targetRangeGrid-targetRange;      % 距离量化误差
maxTargetDelaySample = max(targetDelaySample);       % 最远目标的高采样率时延索引

%% RRC pulse: causal FIR representation used after CP insertion
p = rcosdesign(alpha,span,L,'sqrt').';  % 生成因果有限长RRC FIR脉冲，长度为span*L+1
p = p/norm(p);                          % 将离散脉冲能量归一化为1
pulseMemorySample = length(p)-1;        % FIR滤波器记忆长度，单位为高采样率样本

%% CP is defined before upsampling, hence its high-rate length is L*Ncp
% It must cover both the pulse-shaping memory and the maximum target delay.
NcpFromDesiredRange = ceil(desiredCPRange/symbolRangeSpacing);           % 将40m转换为符号率下的CP样本数
NcpFromTotalMemory = ceil((pulseMemorySample+maxTargetDelaySample)/L);   % 覆盖pulse记忆和最大目标时延所需的CP长度
Ncp = max(NcpFromDesiredRange,NcpFromTotalMemory);% 在符号率序列x上添加的最终CP长度
NcpSample = L*Ncp;                    % 经过L倍上采样后，CP对应的高采样率样本数
Tcp = Ncp*symbolPeriod;               % CP持续时间，单位为s
cpSupportedRange = NcpSample*c*Ts/2;  % CP持续时间对应的单站雷达距离
lengthBlock = L*N;                    % 去CP后pulse-shaped有用波形块的高采样率长度

fprintf('Sampling rate Fs = %.6f GHz\n',Fs/1e9);
fprintf('High-rate range grid = %.6f m/sample\n',rangeSampleSpacing);
fprintf('Symbol-rate range grid = %.6f m/symbol\n',symbolRangeSpacing);
fprintf('RRC pulse memory = %d high-rate samples\n',pulseMemorySample);
fprintf('Maximum target delay = %d high-rate samples\n',maxTargetDelaySample);
fprintf('Ncp = %d symbol-rate samples = %d high-rate samples\n',Ncp,NcpSample);
fprintf('CP duration = %.6f ns\n',Tcp*1e9);
fprintf('CP-supported range = %.6f m\n',cpSupportedRange);

for q = 1:Q
    fprintf('Target %d: nominal range = %.6f m, propagation delay = %.6f ns, delay index = %d, grid range = %.6f m, error = %.3e m\n',q,targetRange(q),targetPropagationDelay(q)*1e9,targetDelaySample(q),targetRangeGrid(q),targetRangeError(q));
end

if NcpSample<pulseMemorySample+maxTargetDelaySample
    error('The CP does not cover the pulse-shaping memory plus maximum target delay.');
end

%% Eq. (35): OFDM modulation basis U = F_N^H
FFTmatrix = exp(-1j*2*pi*(0:N-1).'*(0:N-1)/N)/sqrt(N);% N阶酉DFT矩阵F_N
U = FFTmatrix';      % OFDM调制基U=F_N^H，对频域符号执行归一化IFFT

%% Range axis of the length-LN periodic matched-filter output
delaySample = (0:lengthBlock-1).';       % 周期匹配滤波输出的零基时延索引
rangeAxis = delaySample*c*Ts/2;          % 利用R=k*c*Ts/2将时延索引转换为距离，单位为m
plotIndex = rangeAxis>=0 & rangeAxis<=35;% 只绘制论文所需的0至35m距离区域

%% Preserve the complete Monte Carlo arrays
SimRangeProfile = zeros(Iter,lengthBlock);% 保存每次实验的总含噪平方距离剖面
SimTargetProfile = zeros(Iter,lengthBlock,Q);% 保存每次实验中每个目标的无噪声平方距离剖面

for ii = 1:Iter
    %% Random 16-PSK symbols and OFDM modulation
    symbolIndex = randi([0,Order-1],N,1);% 生成N个独立均匀分布的星座索引
    s = pskmod(symbolIndex,Order,pi/Order);% 生成单位模16-PSK频域通信符号
    x = U*s;% OFDM时域序列x=F_N^H*s，长度为N

    %% Add the CP directly to x before upsampling and pulse shaping
    xCP = [x(end-Ncp+1:end);x];% 先在符号率序列x前复制最后Ncp个样本

    %% Upsample the CP-extended sequence
    xCPUp = complex(zeros(L*length(xCP),1));% 为L倍上采样分配复数零向量
    xCPUp(1:L:end) = xCP;% 每两个有效样本之间插入L-1个零

    %% Physical transmit pulse shaping is a linear convolution
    xTransmit = conv(xCPUp, p,'full');% CP扩展序列与因果RRC脉冲执行物理线性卷积

    %% Select the pulse-shaped useful block after the high-rate CP
    usefulIndex = NcpSample+1:NcpSample+lengthBlock;% 舍弃高采样率CP区间并保留长度LN的有用区间
    xTilde = xTransmit(usefulIndex);% 感知接收机已知的实际有用发射波形模板

    %% Verify Eq. (44)-(47): CP makes useful-block pulse shaping periodic
    if ii==1
        xUp = complex(zeros(lengthBlock,1));% 不含CP的长度LN上采样向量
        xUp(1:L:end) = x;% 对有用OFDM序列x进行L倍上采样
        pPeriodic = complex(zeros(lengthBlock,1));% 将因果RRC脉冲补零到长度LN
        pPeriodic(1:length(p)) = p;% pulse从零时延开始放入周期卷积核
        xTildeEquivalent = ifft(fft(xUp).*fft(pPeriodic));% 论文式(44)-(47)对应的周期pulse shaping
        pulseCircularizationError = norm(xTilde-xTildeEquivalent)/max(norm(xTildeEquivalent),eps);
        fprintf('Pulse-shaping circularization error = %.3e\n',pulseCircularizationError);
    end

    %% Physical sensing channel: each target produces a linear delay
    lengthYSensing = length(xTransmit)+maxTargetDelaySample;% 为最远目标的完整线性时延保留空间
    ysTargetFull = complex(zeros(lengthYSensing,Q));% 第q列保存第q个目标单独产生的完整回波

    for q = 1:Q
        startIndex = targetDelaySample(q)+1;% MATLAB索引等于零基时延索引加1
        endIndex = targetDelaySample(q)+length(xTransmit);% 保持目标回波长度与xTransmit一致
        ysTargetFull(startIndex:endIndex,q) = gamma(q)*xTransmit;% 目标仅对共同发射波形施加复增益和物理线性时延
    end

    ysNoiselessFull = sum(ysTargetFull,2);% 在复数信号层面相干叠加三个目标回波

    %% Select the same useful observation interval at the sensing receiver
    ysTarget = ysTargetFull(usefulIndex,:);% 分别截取三个目标的有用区间，用于画独立目标曲线

    %% Verify that CP converts every physical linear delay to a circular shift
    if ii==1
        for q = 1:Q
            ysTargetEquivalent = gamma(q)*circshift(xTilde,targetDelaySample(q));% CP循环信道的等效目标回波
            targetCircularizationError = norm(ysTarget(:,q)-ysTargetEquivalent)/max(norm(ysTargetEquivalent),eps);
            fprintf('Target %d circularization error = %.3e\n',q,targetCircularizationError);
        end
    end

    %% Add complex Gaussian noise to the selected useful sensing block
    ysNoiseless = ysNoiselessFull(usefulIndex);% 截取总无噪声回波的长度LN有用区间
    signalPower = mean(abs(ysNoiseless).^2);% 三个目标叠加后的有用区间平均功率
    noisePower = signalPower/10^(SNRdB/10);% 根据总输入回波SNR确定复噪声功率
    noise = sqrt(noisePower/2)*(randn(lengthBlock,1)+1j*randn(lengthBlock,1));% 生成CN(0,noisePower)白噪声
    ys = ysNoiseless+noise;% 感知接收机的总含噪有用块

    %% Periodic sensing matched filtering using the actual transmitted block
    for q = 1:Q
        yMatchedTarget = ifft(fft(ysTarget(:,q)).*conj(fft(xTilde)));% 第q个目标回波与共同发射波形的周期互相关
        SimTargetProfile(ii,:,q) = abs(yMatchedTarget.').^2;% 保存第q个目标的平方匹配滤波幅度
    end

    yMatched = ifft(fft(ys).*conj(fft(xTilde)));% 总含噪回波与共同发射波形的周期互相关
    SimRangeProfile(ii,:) = abs(yMatched.').^2;% 保存本次随机通信数据对应的总平方距离剖面
end

%% Ensemble average of the squared matched-filter output
AveRangeProfile = mean(SimRangeProfile,1);% 对随机通信符号实现进行集合平均
AveTargetProfile = squeeze(mean(SimTargetProfile,1));% 对每个独立目标的平方响应进行集合平均
normalization = max(AveRangeProfile);% 所有曲线统一使用总距离剖面的最大值归一化
rangeProfile_dB = 10*log10(AveRangeProfile/normalization+eps);% 功率量转换为dB使用10log10
targetProfile_dB = 10*log10(AveTargetProfile/normalization+eps);% 保留各目标之间的真实相对幅度

%% Plot the three-target range profile
width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

figure(1);
set(gcf,'Units','inches');
set(gcf,'Color','white');
set(gcf,'Renderer','painters');
set(gcf,'PaperUnits','inches');
set(gcf,'PaperPosition',[0,0,width,height]);
set(gcf,'PaperSize',[width,height]);

plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,1),':','Color','#F65314','LineWidth',linewidth);
hold on;
plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,2),':','Color','#00A1F1','LineWidth',linewidth);
plot(rangeAxis(plotIndex),targetProfile_dB(plotIndex,3),':','Color','#8A2BE2','LineWidth',linewidth);
plot(rangeAxis(plotIndex),rangeProfile_dB(plotIndex),'-','Color','#A9A9A9','LineWidth',1.5);

set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend('Target 1','Target 2','Target 3','Range Profile','Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northwest');
labelsize = 16;
xlabel('Range [m]','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Amplitude [dB]','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([0,35]);
ylim([-80,0]);
xticks(0:5:35);
yticks(-80:10:0);
grid on;
set(gca,'GridLineStyle','--','Gridalpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);

% print(gcf,'Fig7_JSAC_CP.pdf','-dpdf','-vector');