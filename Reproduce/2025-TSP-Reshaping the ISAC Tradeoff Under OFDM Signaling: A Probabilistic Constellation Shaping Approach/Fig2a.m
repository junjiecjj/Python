%% Fig. 2(a)：64-PSK 与 64-QAM 的平均模糊函数
% 论文式(1)、(4)、(40)：用采样波形计算模糊函数，并对 5000 次 |Lambda|^2 求平均。
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
B = L*deltaF;
Ts = 1/B;
NMC = 5000;
Nfft = 128;
seed = 42;
rng(seed,'twister');

% 论文式(1)：用 MATLAB 自带调制函数产生单位平均功率的均匀星座符号。
xPSK = pskmod(randi([0,63],L,NMC),64);
xQAM = qammod(randi([0,63],L,NMC),64,'UnitAveragePower',true);
qamAlphabet = qammod((0:63).',64,'UnitAveragePower',true);
fprintf('64-QAM：E{|x|^2}=%.6f，E{|x|^4}=%.6f\n',mean(abs(qamAlphabet).^2),mean(abs(qamAlphabet).^4));

% 论文式(1)：在每个采样区间中点对连续时间子载波和式采样；ifft 仅用于快速计算该和式。
subcarrier = (0:L-1).';
sampleTime = ((0:L-1).'+0.5)*Ts;
midpointPhase = exp(1i*pi*subcarrier/L);
xShiftPSK = bsxfun(@times,xPSK,midpointPhase);
xShiftQAM = bsxfun(@times,xQAM,midpointPhase);
sPSK = L*ifft(xShiftPSK,L,1);
sQAM = L*ifft(xShiftQAM,L,1);

% 多普勒按带宽 B 归一化；长度 Nfft 的 FFT 在一个周期内给出 Nfft 个多普勒点。
delayNormalized = linspace(-1,1,201);
dopplerBins = -Nfft:Nfft;
dopplerNormalized = dopplerBins/Nfft;
[delayGrid,dopplerGrid] = meshgrid(delayNormalized,dopplerNormalized);
delayAxis = Tp*delayNormalized;
dopplerIndex = mod(dopplerBins,Nfft)+1;
averagePowerPSK = zeros(numel(dopplerBins),numel(delayAxis));
averagePowerQAM = zeros(numel(dopplerBins),numel(delayAxis));

for itau = 1:numel(delayAxis)
    tau = delayAxis(itau);
    % 论文式(4)：只在 s(t) 与 s(t-tau) 的共同支撑区间计算相关。
    overlap = sampleTime >= max(0,tau) & sampleTime < min(Tp,Tp+tau);
    if ~any(overlap)
        continue;
    end
    % 论文式(1)：时间延迟在第 l 个子载波上产生 exp(-j2*pi*l*deltaF*tau)。
    delayPhase = exp(-1i*2*pi*subcarrier*deltaF*tau);
    delayedPSK = L*ifft(bsxfun(@times,xShiftPSK,delayPhase),L,1);
    delayedQAM = L*ifft(bsxfun(@times,xShiftQAM,delayPhase),L,1);
    productPSK = bsxfun(@times,sPSK.*conj(delayedPSK),overlap);
    productQAM = bsxfun(@times,sQAM.*conj(delayedQAM),overlap);
    % 论文式(4)、(40)：先对快时间求模糊函数，再对随机实现的功率求平均。
    AFpsk = Ts*fft(productPSK,Nfft,1);
    AFqam = Ts*fft(productQAM,Nfft,1);
    averagePowerPSK(:,itau) = mean(abs(AFpsk(dopplerIndex,:)).^2,2);
    averagePowerQAM(:,itau) = mean(abs(AFqam(dopplerIndex,:)).^2,2);
end

% 论文式(40)：平均功率分别按峰值归一化，显示范围为 -40 至 0 dB。
averageAFpsk = 10*log10(max(averagePowerPSK/max(averagePowerPSK(:)),1e-12));
averageAFqam = 10*log10(max(averagePowerQAM/max(averagePowerQAM(:)),1e-12));
centerDelay = (numel(delayAxis)+1)/2;
centerDoppler = Nfft+1;
assert(max(abs(averagePowerPSK([1,centerDoppler,end],centerDelay)-averagePowerPSK(centerDoppler,centerDelay)))<1e-8,'多普勒周期峰核对未通过。');
fprintf('三个周期峰位于 nu/B=-1、0、1；归一化高度均为 %.2f dB。\n',averageAFpsk(centerDoppler,centerDelay));

% 图2(a)：用稀疏网格同时显示两种星座的三维平均模糊函数。
figure(2);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
rowIndex = 1:2:numel(dopplerBins);
colIndex = 1:2:numel(delayAxis);
hPSK = mesh(delayGrid(rowIndex,colIndex),dopplerGrid(rowIndex,colIndex),max(averageAFpsk(rowIndex,colIndex),-40),'FaceColor','none','EdgeColor','#00A1F1','LineWidth',0.45);
hold on;
hQAM = mesh(delayGrid(rowIndex,colIndex),dopplerGrid(rowIndex,colIndex),max(averageAFqam(rowIndex,colIndex),-40),'FaceColor','none','EdgeColor','#F65314','LineWidth',0.45);
hold off;
set(gca,'FontName','Times New Roman','FontSize',fontsize,'LineWidth',1);
xlabel('Normalized Delay','FontName','Times New Roman','FontSize',fontsize);
ylabel('Normalized Doppler','FontName','Times New Roman','FontSize',fontsize);
zlabel('Average AF (dB)','FontName','Times New Roman','FontSize',fontsize);
legend([hPSK,hQAM],{'64-PSK','64-QAM'},'FontName','Times New Roman','FontSize',12,'Location','northeast');
xlim([-1,1]);
ylim([-1,1]);
zlim([-40,0]);
xticks(-1:0.5:1);
yticks(-1:0.5:1);
zticks(-40:10:0);
view(-48,26);
grid on;
box on;
drawnow;
print(gcf,'Fig2a_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig2a_2024_TSP.pdf','-dpdf','-vector');