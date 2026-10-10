%% Fig. 2(c)：64-PSK 单次实现的零多普勒切片
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
seed = 42;
rng(seed,'twister');

% 论文式(1)：生成一次均匀分布、单位平均功率的 64-PSK 星座符号。
xPSK = pskmod(randi([0,63],L,1),64);

% 论文式(4)、图2(c)：沿用图2(b)的时延格点，设置多普勒为零。
delayNormalized = (-L+1:L-1)/L;
delayAxis = Tp*delayNormalized;
subcarrier = (0:L-1).';
differenceIndex = bsxfun(@minus,subcarrier,subcarrier.');
symbolProduct = xPSK*xPSK';
AFpsk = zeros(size(delayAxis));

for itau = 1:numel(delayAxis)
    tau = delayAxis(itau);
    Tmin = max(0,tau);
    Tmax = min(Tp,Tp+tau);
    Tdiff = Tmax-Tmin;
    Tavg = (Tmax+Tmin)/2;
    % 论文式(5)：计算各子载波对在共同支撑区间上的精确积分，包含自项与交叉项。
    sincArgument = differenceIndex*deltaF*Tdiff;
    sincValue = ones(size(sincArgument));
    nonzeroIndex = abs(sincArgument)>1e-12;
    sincValue(nonzeroIndex) = sin(pi*sincArgument(nonzeroIndex))./(pi*sincArgument(nonzeroIndex));
    integralWeight = Tdiff*sincValue.*exp(1i*2*pi*differenceIndex*deltaF*Tavg);
    delayPhase = exp(1i*2*pi*subcarrier*deltaF*tau);
    AFpsk(itau) = sum(sum(symbolProduct.*bsxfun(@times,integralWeight,delayPhase.')));
end

% 论文图2(c)：对这一次实现的模糊函数取模平方，按自身峰值归一化，不做随机平均。
powerPSK = abs(AFpsk).^2;
slicePSK = 10*log10(max(powerPSK/max(powerPSK),1e-12));
[~,zeroIndex] = min(abs(delayNormalized));
assert(abs(slicePSK(zeroIndex))<1e-8,'零时延峰值归一化未通过。');
assert(max(abs(powerPSK-fliplr(powerPSK)))<1e-8*max(powerPSK),'零多普勒切片的功率对称性核对未通过。');

% 论文图2(c)：64-PSK 使用蓝色实线，无 marker。
figure(2);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
plot(delayNormalized,slicePSK,'-','Color','#00A1F1','LineWidth',linewidth);
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend({'64-PSK'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northeast');
labelsize = 16;
xlabel('Normalized Delay','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Zero Doppler Slice (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-1,1]);
ylim([-40,0]);
xticks(-1:0.5:1);
yticks(-40:5:0);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig2c_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig2c_2024_TSP.pdf','-dpdf','-vector');