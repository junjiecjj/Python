%% Fig. 7(b)：64-QAM启发式PCS的AIR随c0变化
% 方法复现：64-QAM的P3非唯一，本文采用补充的最小距离选解规则。
% 需要Communications Toolbox及Optimization Toolbox；原文未公布完整概率向量。
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
noiseVariances = [0.005,0.025,0.05];
xQAM = qammod((0:63).',64,'UnitAveragePower',true);
% 论文式(38)、(39)：由单位平均功率64-QAM计算边界；均匀点精确值1.380952，原文标为1.3805。
c0Minimum = 457/441;
c0Uniform = mean(abs(xQAM).^4);
c0Maximum = 143/63;
c0Axis = unique([1:0.1:3,c0Minimum,c0Uniform,c0Maximum]);
AIR = zeros(3,numel(c0Axis));

for ic0 = 1:numel(c0Axis)
    % 论文式(38)、(39)：64-QAM可达四阶矩为[457/441,143/63]；不可达目标取最近边界。
    p = pcs64Probability(xQAM,c0Axis(ic0));
    for inoise = 1:3
        % 论文式(25)、(26)：按给定概率生成AWGN观测，并用蒙特卡罗积分计算AIR。
        AIR(inoise,ic0) = airMonteCarlo(xQAM,p,noiseVariances(inoise),NMC);
    end
end
fprintf('64-QAM：最小四阶矩%.8f，均匀四阶矩%.8f，最大四阶矩%.8f。\n',c0Minimum,c0Uniform,c0Maximum);

% 论文图7(b)：三条曲线依次为实线、点划线、虚线，无marker。
figure(7);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
plot(c0Axis,AIR(1,:),'-','Color','#00A1F1','LineWidth',linewidth);
hold on;
plot(c0Axis,AIR(2,:),'-.','Color','#F65314','LineWidth',linewidth);
plot(c0Axis,AIR(3,:),'--','Color','#FFBB00','LineWidth',linewidth);
hold off;
legendLabels = {'$\sigma^2=0.005$','$\sigma^2=0.025$','$\sigma^2=0.05$'};
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend(legendLabels,'Interpreter','latex');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','northeast');
xlabel('$c_0$','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('Achievable Information Rate (bps/Hz)','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([1,3]);
ylim([2,6.5]);
xticks(1:0.5:3);
yticks(2:0.5:6.5);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig7b_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig7b_2024_TSP.pdf','-dpdf','-vector');

function p = pcs64Probability(x,c0)
    % 论文式(38)、(39)：同圆环内等概率，先求P3能达到的最近四阶矩。
    Q = numel(x);
    [ringPower,~,ringIndex] = unique(round(abs(x).^2,12));
    W = numel(ringPower);
    ringCount = accumarray(ringIndex,1,[W,1]);
    uniformMass = ringCount/Q;
    lowerRing = find(ringPower<1,1,'last');
    upperRing = find(ringPower>1,1,'first');
    minimumMass = zeros(W,1);
    minimumMass(lowerRing) = (ringPower(upperRing)-1)/(ringPower(upperRing)-ringPower(lowerRing));
    minimumMass(upperRing) = 1-minimumMass(lowerRing);
    maximumMass = zeros(W,1);
    maximumMass(1) = (ringPower(end)-1)/(ringPower(end)-ringPower(1));
    maximumMass(end) = 1-maximumMass(1);
    minimumMoment = sum(minimumMass.*ringPower.^2);
    maximumMoment = sum(maximumMass.*ringPower.^2);
    uniformMoment = sum(uniformMass.*ringPower.^2);
    effectiveMoment = min(max(c0,minimumMoment),maximumMoment);
    if effectiveMoment<=minimumMoment+1e-10
        mass = minimumMass;
    elseif effectiveMoment>=maximumMoment-1e-10
        mass = maximumMass;
    elseif abs(effectiveMoment-uniformMoment)<1e-10
        mass = uniformMass;
    else
        % 补充选解规则：P3最优解不唯一；选取与均匀逐点概率欧氏距离最小的解，不使用MBA。
        % 圆环总概率为mass；距离平方为sum((mass-ringCount/Q).^2./ringCount)。
        H = 2*diag(1./ringCount);
        f = -2*ones(W,1)/Q;
        Aeq = [ones(1,W);ringPower.';(ringPower.^2).'];
        beq = [1;1;effectiveMoment];
        options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off','OptimalityTolerance',1e-10,'ConstraintTolerance',1e-10);
        [mass,~,exitflag] = quadprog(H,f,[],[],Aeq,beq,zeros(W,1),ones(W,1),[],options);
        assert(exitflag>0,'PCS概率求解失败。');
        assert(all(mass>=-1e-8),'PCS概率出现明显负值。');
        mass = max(mass,0);
        mass = mass/sum(mass);
    end
    p = mass(ringIndex)./ringCount(ringIndex);
    % 论文式(38)、(39)：检查概率归一化、单位平均功率与P3对应的可达四阶矩。
    assert(abs(sum(p)-1)<1e-7,'概率和不为1。');
    assert(abs(sum(p.*abs(x).^2)-1)<1e-7,'平均功率不为1。');
    assert(abs(sum(p.*abs(x).^4)-effectiveMoment)<1e-7,'四阶矩不满足要求。');
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