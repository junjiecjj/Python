%% Fig. 6(a)：16-QAM启发式PCS的SO-CFAR检测概率
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
seed = 42;
rng(seed,'twister');

L = 64;
Tp = 1;
Ts = Tp/L;
NMC = 5000;
sensingSNRdB = -5:1:10;
Pfa = 1e-4;
targetCell = 8;
selfInterferenceSNRdB = 10;
noiseVariance = 1;

% 补充参数：每侧16个训练单元、2个保护单元，回波系数相位取零；SNR取匹配滤波前参考。
Ntrain = 16;
Nguard = 2;
alphaSO = soCFARFactor(Pfa,Ntrain);
rangeCells = [targetCell,targetCell-Nguard-Ntrain:targetCell-Nguard-1,targetCell+Nguard+1:targetCell+Nguard+Ntrain];

% 论文式(1)、(38)：生成完整星座，求16-QAM启发式PCS概率，不使用MBA。
xPSK = pskmod((0:15).',16);
xQAM = qammod((0:15).',16,'UnitAveragePower',true);
alphabets = {xPSK,xQAM,xQAM,xQAM,xQAM};
probabilities = {ones(16,1)/16,ones(16,1)/16,pcs16Probability(xQAM,1.05),pcs16Probability(xQAM,1.15),pcs16Probability(xQAM,1.25)};
Pd = zeros(5,numel(sensingSNRdB));
Nreceive = 3*L-2;
Nfft = 2^nextpow2(Nreceive+L-1);

for imethod = 1:5
    symbols = drawSymbols(alphabets{imethod},probabilities{imethod},L,NMC);

    % 论文式(1)：在t=n*Ts采样，并对原文波形除以sqrt(L)，使期望发射平均功率为1。
    waveform = sqrt(L)*ifft(symbols,L,1);
    filterFFT = fft(conj(flipud(waveform)),Nfft,1);

    % 论文式(3)、(4)：用Ts乘相关求和近似连续时间匹配滤波积分。
    ACF = Ts*ifft(fft(waveform,Nfft,1).*filterFFT,Nfft,1);

    % 论文式(2)、(3)：接收噪声覆盖n=-L+1至2L-2，保证检测窗内模板完整。
    receiveNoise = sqrt(noiseVariance/2)*(randn(Nreceive,NMC)+1i*randn(Nreceive,NMC));
    matchedNoise = Ts*ifft(fft(receiveNoise,Nfft,1).*filterFFT,Nfft,1);
    selfInterference = sqrt(noiseVariance*10^(selfInterferenceSNRdB/10))*ACF(rangeCells+L,:);
    targetResponse = ACF(rangeCells-targetCell+L,:);
    noiseResponse = matchedNoise(rangeCells+2*L-1,:);

    for iSNR = 1:numel(sensingSNRdB)
        % 论文式(2)、(3)的补充约定：SNR按匹配滤波前的目标平均回波功率与接收噪声功率定义。
        output = selfInterference+sqrt(noiseVariance*10^(sensingSNRdB(iSNR)/10))*targetResponse+noiseResponse;
        outputPower = abs(output).^2;
        noiseLeft = mean(outputPower(2:Ntrain+1,:),1);
        noiseRight = mean(outputPower(Ntrain+2:2*Ntrain+1,:),1);
        threshold = alphaSO*min(noiseLeft,noiseRight);
        Pd(imethod,iSNR) = mean(outputPower(1,:)>threshold);
    end
end
fprintf('SO-CFAR阈值系数为 %.6f；每侧训练单元%d，保护单元%d。\n',alphaSO,Ntrain,Nguard);

% 论文图6(a)：保留原图的线型、marker与曲线顺序。
figure(6);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
lineStyles = {'-o','-s','-','--','-.'};
colors = {'#00A1F1','#7CBB00','#F65314','#F65314','#F65314'};
hold on;
for imethod = 1:5
    plot(sensingSNRdB,Pd(imethod,:),lineStyles{imethod},'Color',colors{imethod},'LineWidth',linewidth,'MarkerSize',markersize);
end
hold off;
legendLabels = {'16-PSK: Uniform Distribution','16-QAM: Uniform Distribution','16-QAM-PCS: $c_0=1.05$','16-QAM-PCS: $c_0=1.15$','16-QAM-PCS: $c_0=1.25$'};
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend(legendLabels,'Interpreter','latex');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','northwest');
xlabel('Sensing SNR (dB)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('$P_{\rm d}$','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([-5,10]);
ylim([0,1]);
xticks(-5:5:10);
yticks(0:0.2:1);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig6a_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig6a_2024_TSP.pdf','-dpdf','-vector');

function p = pcs16Probability(x,c0)
    % 论文式(38)、(39)：16-QAM三个圆环的唯一解；不可达目标取最近的可达四阶矩。
    effectiveMoment = min(max(c0,1),1.64);
    massInner = (effectiveMoment-1)/1.28;
    massMiddle = 1-2*massInner;
    ringPower = round(abs(x).^2,12);
    p = zeros(size(x));
    p(ringPower==0.2) = massInner/4;
    p(ringPower==1) = massMiddle/8;
    p(ringPower==1.8) = massInner/4;
end

function symbols = drawSymbols(alphabet,p,nRows,nColumns)
    % 按指定先验概率抽取星座索引，星座本身由MATLAB调制函数产生。
    u = rand(nRows*nColumns,1);
    cdf = cumsum(p);
    cdf(end) = 1;
    index = ones(size(u));
    for iq = 1:numel(alphabet)-1
        index = index+(u>cdf(iq));
    end
    symbols = reshape(alphabet(index),nRows,nColumns);
end

function alpha = soCFARFactor(Pfa,Ntrain)
    % SO-CFAR补充推导：独立指数噪声下，两侧训练均值服从Gamma分布。
    % 由Pfa=E{exp[-alpha*min(Zleft,Zright)]}求阈值系数，不使用CA-CFAR的系数。
    k = (0:Ntrain-1).';
    logCoefficient = gammaln(Ntrain+k)-gammaln(Ntrain)-gammaln(k+1);
    probability = @(a) 2*sum(exp(logCoefficient+(Ntrain+k)*log(Ntrain/(2*Ntrain+a))));
    upper = 1;
    while probability(upper)>Pfa
        upper = 2*upper;
    end
    alpha = fzero(@(a) probability(a)-Pfa,[0,upper]);
end