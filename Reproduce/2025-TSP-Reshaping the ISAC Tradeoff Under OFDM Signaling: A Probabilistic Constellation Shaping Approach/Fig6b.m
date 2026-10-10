%% Fig. 6(b)：16-QAM启发式PCS的AIR随c0变化
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
c0Axis = unique([1:0.05:2.2,1.32,1.64]);
noiseVariances = [0.01,0.05,0.1];
xQAM = qammod((0:15).',16,'UnitAveragePower',true);
AIR = zeros(3,numel(c0Axis));

for ic0 = 1:numel(c0Axis)
    % 论文式(38)、(39)：16-QAM可达四阶矩为[1,1.64]；超过上界后概率保持不变。
    p = pcs16Probability(xQAM,c0Axis(ic0));
    for inoise = 1:3
        % 论文式(25)、(26)：按给定概率生成AWGN观测，并用蒙特卡罗积分计算AIR。
        AIR(inoise,ic0) = airMonteCarlo(xQAM,p,noiseVariances(inoise),NMC);
    end
end
fprintf('均匀16-QAM：四阶矩1.32、输入熵4 bit；最大四阶矩1.64时输入熵3 bit。\n');

% 论文图6(b)：三条曲线依次为实线、点划线、虚线，无marker。
figure(6);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
plot(c0Axis,AIR(1,:),'-','Color','#00A1F1','LineWidth',linewidth);
hold on;
plot(c0Axis,AIR(2,:),'-.','Color','#F65314','LineWidth',linewidth);
plot(c0Axis,AIR(3,:),'--','Color','#FFBB00','LineWidth',linewidth);
hold off;
legendLabels = {'$\sigma^2=0.01$','$\sigma^2=0.05$','$\sigma^2=0.1$'};
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend(legendLabels,'Interpreter','latex');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','northeast');
xlabel('$c_0$','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('Achievable Information Rate (bps/Hz)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([1,2.2]);
ylim([2.6,4.2]);
xticks(1:0.2:2.2);
yticks(2.6:0.2:4.2);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig6b_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig6b_2024_TSP.pdf','-dpdf','-vector');

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