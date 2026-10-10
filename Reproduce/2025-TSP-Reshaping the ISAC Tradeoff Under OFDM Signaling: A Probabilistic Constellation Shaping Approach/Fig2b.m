%% Fig. 2(b)：64-PSK 与 64-QAM 的平均零多普勒切片
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

% 论文式(1)：用 MATLAB 自带调制函数产生均匀分布、单位平均功率的星座符号。
xPSK = pskmod(randi([0,63],L,NMC),64);
xQAM = qammod(randi([0,63],L,NMC),64,'UnitAveragePower',true);
qamAlphabet = qammod((0:63).',64,'UnitAveragePower',true);
fprintf('64-QAM：E{|x|^2}=%.6f，E{|x|^4}=%.6f\n',mean(abs(qamAlphabet).^2),mean(abs(qamAlphabet).^4));

% 论文式(4)：在归一化时延的全区间取点，并加密主瓣及其附近的零点。
delayNormalized = (-L+1:L-1)/L;
delayAxis = Tp*delayNormalized;
powerPSK = zeros(size(delayAxis));
powerQAM = zeros(size(delayAxis));
subcarrier = (0:L-1).';
differenceIndex = (-(L-1):L-1).';
Nconv = 2^nextpow2(2*L-1);
fftPSK = fft(xPSK,Nconv,1);
fftQAM = fft(xQAM,Nconv,1);

for itau = 1:numel(delayAxis)
    tau = delayAxis(itau);
    Tmin = max(0,tau);
    Tmax = min(Tp,Tp+tau);
    Tdiff = Tmax-Tmin;
    if Tdiff <= 0
        continue;
    end
    Tavg = (Tmax+Tmin)/2;
    % 论文式(1)、(4)、(5)：先对矩形窗共同支撑区间精确积分，再按子载波差分下标合并求和。
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
    % 论文式(40)：先对每次试验的模糊函数取模平方，再对 5000 次试验求平均。
    powerPSK(itau) = mean(abs(AFpsk).^2);
    powerQAM(itau) = mean(abs(AFqam).^2);
end

% 论文式(40)：两条平均零多普勒切片分别按各自峰值归一化，并转换为 dB。
averagePSK = 10*log10(max(powerPSK/max(powerPSK),1e-12));
averageQAM = 10*log10(max(powerQAM/max(powerQAM),1e-12));
[~,zeroIndex] = min(abs(delayNormalized));
[~,nullIndex] = min(abs(delayNormalized-1/L));
assert(abs(averagePSK(zeroIndex))<1e-8 && abs(averageQAM(zeroIndex))<1e-8,'零时延峰值归一化未通过。');
fprintf('tau/Tp=1/L 附近：64-PSK 为 %.2f dB，64-QAM 为 %.2f dB，相差 %.2f dB。\n',averagePSK(nullIndex),averageQAM(nullIndex),averageQAM(nullIndex)-averagePSK(nullIndex));

% 论文图 2(b)：蓝色实线为 64-PSK，红色实线为 64-QAM，无 marker。
figure(2);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
hPSK = plot(delayNormalized,averagePSK,'-','Color','#00A1F1','LineWidth',linewidth);
hold on;
hQAM = plot(delayNormalized,averageQAM,'-','Color','#F65314','LineWidth',linewidth);
hold off;
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend([hPSK,hQAM],{'64-PSK','64-QAM'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','northeast');
labelsize = 16;
xlabel('Normalized Delay','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
ylabel('Zero Doppler Slice (dB)','FontSize',labelsize,'FontName','Times New Roman','Interpreter','latex');
xlim([-1,1]);
ylim([-40,0]);
xticks(-1:0.5:1);
yticks(-40:10:0);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig2b_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig2b_2024_TSP.pdf','-dpdf','-vector');