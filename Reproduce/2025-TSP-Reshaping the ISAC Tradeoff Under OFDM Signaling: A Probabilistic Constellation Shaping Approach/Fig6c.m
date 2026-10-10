%% Fig. 6(c)：16-QAM启发式PCS的AIR随通信SNR变化
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

NMC = 5000;
communicationSNRdB = 0:2.5:25;

% 论文式(1)、(38)：生成均匀PSK/QAM星座，以及三组启发式PCS概率。
xPSK = pskmod((0:15).',16);
xQAM = qammod((0:15).',16,'UnitAveragePower',true);
alphabets = {xPSK,xQAM,xQAM,xQAM,xQAM};
probabilities = {ones(16,1)/16,ones(16,1)/16,pcs16Probability(xQAM,1.05),pcs16Probability(xQAM,1.15),pcs16Probability(xQAM,1.25)};
AIR = zeros(5,numel(communicationSNRdB));

for imethod = 1:5
    p = probabilities{imethod};
    positiveProbability = p(p>0);
    inputEntropy = -sum(positiveProbability.*log2(positiveProbability));
    fprintf('第%d条曲线的高SNR极限为 %.6f bit。\n',imethod,inputEntropy);
    for iSNR = 1:numel(communicationSNRdB)
        % 论文式(24)：单位平均输入功率下，复高斯噪声方差为10^(-SNR/10)。
        noiseVariance = 10^(-communicationSNRdB(iSNR)/10);
        % 论文式(25)、(26)：计算单个子载波的AIR，不再乘子载波数量L。
        AIR(imethod,iSNR) = airMonteCarlo(alphabets{imethod},p,noiseVariance,NMC);
    end
end

% 论文图6(c)：保留原图的线型、marker与曲线顺序。
figure(6);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
lineStyles = {'-o','-s','-','--','-.'};
colors = {'#00A1F1','#7CBB00','#F65314','#F65314','#F65314'};
hold on;
for imethod = 1:5
    plot(communicationSNRdB,AIR(imethod,:),lineStyles{imethod},'Color',colors{imethod},'LineWidth',linewidth,'MarkerSize',markersize);
end
hold off;
legendLabels = {'16-PSK: Uniform Distribution','16-QAM: Uniform Distribution','16-QAM-PCS: $c_0=1.05$','16-QAM-PCS: $c_0=1.15$','16-QAM-PCS: $c_0=1.25$'};
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend(legendLabels,'Interpreter','latex');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','southeast');
xlabel('SNR (dB)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('Achievable Information Rate (bps/Hz)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([0,25]);
ylim([0.5,4.5]);
xticks(0:5:25);
yticks(0.5:0.5:4.5);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig6c_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig6c_2024_TSP.pdf','-dpdf','-vector');

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

function AIR = airMonteCarlo(alphabet,p,noiseVariance,NMC)
    % 论文式(24)、(26)：y=x+n，复高斯噪声满足E{|n|^2}=noiseVariance。
    x = drawSymbols(alphabet,p,NMC,1);
    noise = sqrt(noiseVariance/2)*(randn(NMC,1)+1i*randn(NMC,1));
    y = x+noise;

    % 论文式(25)、(26)：用log-sum-exp计算高斯混合密度，避免高SNR下数值下溢。
    logTerms = bsxfun(@plus,-abs(bsxfun(@minus,y,alphabet.')).^2/noiseVariance,log(p.'));
    rowMaximum = max(logTerms,[],2);
    logMixture = rowMaximum+log(sum(exp(bsxfun(@minus,logTerms,rowMaximum)),2));

    % 论文式(25)、(26)：H(Y)-log(pi*e*noiseVariance)，换算为单子载波bps/Hz。
    AIR = (-mean(logMixture)-1)/log(2);
end