%% Fig. 3：总平均模糊函数的仿真结果，以及自项、交叉项的解析结果
clear;
clc;
close all;

width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');

L = 64;
Tp = 1;
deltaF = 1/Tp;
NMC = 5000;
seed = 42;
rng(seed,'twister');

% 论文式(1)、(11)：生成单位平均功率星座，并从完整星座精确计算四阶矩。
xPSK = pskmod(randi([0,63],L,NMC),64);
xQAM = qammod(randi([0,63],L,NMC),64,'UnitAveragePower',true);
qamAlphabet = qammod((0:63).',64,'UnitAveragePower',true);
kappaPSK = 1;
kappaQAM = mean(abs(qamAlphabet).^4);

% 图3复现取点：正文未指定网格；采用含端点的127点网格以呈现原图的PSK自项虚线。
delayNormalized = linspace(-1,1,2*L-1);
delayAxis = Tp*delayNormalized;
subcarrier = (0:L-1).';
differenceIndex = (-(L-1):L-1).';
positiveDifference = (1:L-1).';
Nconv = 2^nextpow2(2*L-1);
fftPSK = fft(xPSK,Nconv,1);
fftQAM = fft(xQAM,Nconv,1);
powerPSK = zeros(size(delayAxis));
powerQAM = zeros(size(delayAxis));
selfPSK = zeros(size(delayAxis));
selfQAM = zeros(size(delayAxis));
crossPower = zeros(size(delayAxis));

for itau = 1:numel(delayAxis)
    tau = delayAxis(itau);
    Tmin = max(0,tau);
    Tmax = min(Tp,Tp+tau);
    Tdiff = Tmax-Tmin;
    if Tdiff <= 0
        continue;
    end
    Tavg = (Tmax+Tmin)/2;

    % 论文式(5)：精确积分连续时间模糊函数；FFT仅用于加速子载波差分下标的求和。
    sincArgument = differenceIndex*deltaF*Tdiff;
    sincValue = ones(size(sincArgument));
    nonzeroIndex = abs(sincArgument)>1e-12;
    sincValue(nonzeroIndex) = sin(pi*sincArgument(nonzeroIndex))./(pi*sincArgument(nonzeroIndex));
    integralWeight = Tdiff*sincValue.*exp(1i*2*pi*differenceIndex*deltaF*Tavg);
    delayPhase = exp(1i*2*pi*subcarrier*deltaF*tau);
    correlationPSK = ifft(fftPSK.*fft(flipud(bsxfun(@times,conj(xPSK),delayPhase)),Nconv,1),Nconv,1);
    correlationQAM = ifft(fftQAM.*fft(flipud(bsxfun(@times,conj(xQAM),delayPhase)),Nconv,1),Nconv,1);
    AFpsk = integralWeight.'*correlationPSK(1:2*L-1,:);
    AFqam = integralWeight.'*correlationQAM(1:2*L-1,:);

    % 论文式(40)：对5000次随机实现的模糊函数功率求平均。
    powerPSK(itau) = mean(abs(AFpsk).^2);
    powerQAM(itau) = mean(abs(AFqam).^2);

    % 论文式(5)、(11)：自项平均功率等于其均值模平方与方差之和。
    meanSelf = Tdiff*sum(exp(1i*2*pi*subcarrier*deltaF*tau));
    selfPSK(itau) = abs(meanSelf)^2+L*Tdiff^2*(kappaPSK-1);
    selfQAM(itau) = abs(meanSelf)^2+L*Tdiff^2*(kappaQAM-1);

    % 论文式(14)：差分为正负k的子载波对各有L-k个，交叉项平均功率与星座无关。
    crossSinc = sin(pi*positiveDifference*deltaF*Tdiff)./(pi*positiveDifference*deltaF*Tdiff);
    crossPower(itau) = 2*Tdiff^2*sum((L-positiveDifference).*crossSinc.^2);
end

% 论文式(8)、(41)：总解析平均功率等于自项与交叉项平均功率之和。
totalAnalyticPSK = selfPSK+crossPower;
totalAnalyticQAM = selfQAM+crossPower;
[~,zeroIndex] = min(abs(delayNormalized));
normalizerPSK = totalAnalyticPSK(zeroIndex);
normalizerQAM = totalAnalyticQAM(zeroIndex);
floorPower = 1e-12;
simulationPSK = 10*log10(max(powerPSK/normalizerPSK,floorPower));
simulationQAM = 10*log10(max(powerQAM/normalizerQAM,floorPower));
analyticSelfPSK = 10*log10(max(selfPSK/normalizerPSK,floorPower));
analyticSelfQAM = 10*log10(max(selfQAM/normalizerQAM,floorPower));
analyticCrossPSK = 10*log10(max(crossPower/normalizerPSK,floorPower));
analyticCrossQAM = 10*log10(max(crossPower/normalizerQAM,floorPower));

% 论文式(41)：在功率域核对总解析结果与蒙特卡罗结果，避免直接相加dB曲线。
checkIndex = abs(delayNormalized)>1e-12 & abs(delayNormalized)<1;
errorPSK = norm(powerPSK(checkIndex)-totalAnalyticPSK(checkIndex))/norm(totalAnalyticPSK(checkIndex));
errorQAM = norm(powerQAM(checkIndex)-totalAnalyticQAM(checkIndex))/norm(totalAnalyticQAM(checkIndex));
fprintf('64-QAM四阶矩为 %.6f。\n',kappaQAM);
fprintf('旁瓣平均功率的相对误差：PSK %.4f%%，QAM %.4f%%。\n',100*errorPSK,100*errorQAM);

% 论文图3：实线为总仿真功率，虚线为解析自项功率，点线为解析交叉项功率。
figure(3);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
h1 = plot(delayNormalized,simulationPSK,'-','Color','#00A1F1','LineWidth',linewidth);
hold on;
h2 = plot(delayNormalized,simulationQAM,'-','Color','#F65314','LineWidth',linewidth);
h3 = plot(delayNormalized,analyticSelfPSK,'--','Color','#00A1F1','LineWidth',linewidth);
h4 = plot(delayNormalized,analyticSelfQAM,'--','Color','#F65314','LineWidth',linewidth);
h5 = plot(delayNormalized,analyticCrossPSK,':','Color','#00A1F1','LineWidth',linewidth);
h6 = plot(delayNormalized,analyticCrossQAM,':','Color','#F65314','LineWidth',linewidth);
hold off;
set(gca,'FontSize',16,'FontName','Times New Roman');
legendLabels = {'64-PSK: $\overline{\Lambda}(\tau,0)$, simulation','64-QAM: $\overline{\Lambda}(\tau,0)$, simulation','64-PSK: $\overline{\Lambda}_{S}(\tau,0)$, analytical','64-QAM: $\overline{\Lambda}_{S}(\tau,0)$, analytical','64-PSK: $\overline{\Lambda}_{C}(\tau,0)$, analytical','64-QAM: $\overline{\Lambda}_{C}(\tau,0)$, analytical'};
h_legend = legend([h1,h2,h3,h4,h5,h6],legendLabels,'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','southeast');
labelsize = 16;
xlabel('Normalized Delay','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Zero Doppler Slice (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-1,1]);
ylim([-70,0]);
xticks(-1:0.5:1);
yticks(-70:10:0);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig3_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig3_2024_TSP.pdf','-dpdf','-vector');