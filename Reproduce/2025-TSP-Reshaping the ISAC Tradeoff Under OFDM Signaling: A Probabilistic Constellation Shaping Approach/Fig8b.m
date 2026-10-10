%% Fig. 8(b)：分别检测MBA与启发式PCS的随机OFDM回波
% 原文未明确给出此图的固定感知SNR和检测窗；以下补充参数不代表作者的原始设置。
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

if isfile('Fig8_PCS_results.mat')
    cached = load('Fig8_PCS_results.mat','result');
    result = cached.result;
    assert(isfield(result,'schema') && result.schema==1,'概率缓存版本不匹配，请重新运行图8(a)。');
else
    result = Fig8_PCS_MBA;
    save('Fig8_PCS_results.mat','result');
end
assert(isequal(result.cfg.noiseVariances,[0.005,0.01]),'此绘图脚本要求两种通信噪声方差为0.005和0.01。');

L = 64;
Tp = 1;
Ts = Tp/L;
NMC = 5000;
Pfa = 1e-4;
targetCell = 8;
selfInterferenceSNRdB = 10;
% 补充约定：固定感知SNR取0 dB，噪声取匹配滤波前离散接收噪声，与图6、7代码定义一致。
sensingSNRdB = 0;
noiseVariance = 1;
Ntrain = 16;
Nguard = 2;
alphaSO = soCFARFactor(Pfa,Ntrain);
rangeCells = [targetCell,targetCell-Nguard-Ntrain:targetCell-Nguard-1,targetCell+Nguard+1:targetCell+Nguard+Ntrain];
Nreceive = 3*L-2;
Nfft = 2^nextpow2(Nreceive+L-1);
nC0 = numel(result.cfg.c0Axis);
Pd = zeros(4,nC0);
PdStandardError = zeros(4,nC0);
pairedDifference = zeros(2,nC0);
pairedStandardError = zeros(2,nC0);

for ic0 = 1:nC0
    rng(42+ic0,'twister');
    % 补充方差降低：四条曲线共享符号抽样的均匀随机数和接收噪声，分别使用自己的概率分布。
    uniformSamples = rand(L,NMC);
    receiveNoise = sqrt(noiseVariance/2)*(randn(Nreceive,NMC)+1i*randn(Nreceive,NMC));
    probabilities = {result.pOptimal(:,ic0,1),result.pHeuristic(:,ic0),result.pOptimal(:,ic0,2),result.pHeuristic(:,ic0)};
    decisions = false(4,NMC);
    for icurve = 1:4
        % 论文式(1)、(31)～(38)：星座由qammod生成，按对应MBA或启发式概率抽样。
        symbols = drawWithUniform(result.x,probabilities{icurve},uniformSamples);
        waveform = sqrt(L)*ifft(symbols,L,1);
        filterFFT = fft(conj(flipud(waveform)),Nfft,1);
        % 论文式(3)、(4)：线性匹配滤波；Ts将离散相关求和换算为连续积分的采样近似。
        ACF = Ts*ifft(fft(waveform,Nfft,1).*filterFFT,Nfft,1);
        matchedNoise = Ts*ifft(fft(receiveNoise,Nfft,1).*filterFFT,Nfft,1);
        selfInterference = sqrt(noiseVariance*10^(selfInterferenceSNRdB/10))*ACF(rangeCells+L,:);
        targetResponse = sqrt(noiseVariance*10^(sensingSNRdB/10))*ACF(rangeCells-targetCell+L,:);
        noiseResponse = matchedNoise(rangeCells+2*L-1,:);
        % 论文式(2)、(3)：SI位于0号距离单元，弱目标位于8号距离单元。
        outputPower = abs(selfInterference+targetResponse+noiseResponse).^2;
        noiseLeft = mean(outputPower(2:Ntrain+1,:),1);
        noiseRight = mean(outputPower(Ntrain+2:2*Ntrain+1,:),1);
        threshold = alphaSO*min(noiseLeft,noiseRight);
        decisions(icurve,:) = outputPower(1,:)>threshold;
        Pd(icurve,ic0) = mean(decisions(icurve,:));
        PdStandardError(icurve,ic0) = sqrt(Pd(icurve,ic0)*(1-Pd(icurve,ic0))/NMC);
    end
    % 不能由四阶矩相同直接设定Pd相同，记录实际配对检测差与标准误差。
    for inoise = 1:2
        difference = double(decisions(2*inoise-1,:))-double(decisions(2*inoise,:));
        pairedDifference(inoise,ic0) = mean(difference);
        pairedStandardError(inoise,ic0) = std(difference)/sqrt(NMC);
    end
    fprintf('c0=%.3f：Pd=[%.4f %.4f %.4f %.4f]。\n',result.cfg.c0Axis(ic0),Pd(:,ic0));
end
fprintf('固定感知SNR=%.2f dB，SO-CFAR名义Pfa=%.1e，阈值系数%.6f。\n',sensingSNRdB,Pfa,alphaSO);
save('Fig8_detection_results.mat','Pd','PdStandardError','pairedDifference','pairedStandardError','sensingSNRdB','noiseVariance','Ntrain','Nguard','Pfa','alphaSO','result');

% 论文图8(b)：通信噪声方差只决定概率设计；四条曲线使用同一个感知噪声与SNR。
figure(8);
set(gcf,'Units','inches','Position',[1,1,width,height]);
set(gcf,'Color','white','Renderer','painters');
set(gcf,'PaperUnits','inches','PaperPosition',[0,0,width,height],'PaperSize',[width,height]);
lineStyles = {'-o','-o','--o','--o'};
colors = {'#F65314','#00A1F1','#F65314','#00A1F1'};
hold on;
for icurve = 1:4
    plot(result.cfg.c0Axis,Pd(icurve,:),lineStyles{icurve},'Color',colors{icurve},'LineWidth',linewidth,'MarkerSize',markersize);
end
hold off;
legendLabels = {'Optimal PCS, $\sigma^2=0.005$','Heuristic PCS, $\sigma^2=0.005$','Optimal PCS, $\sigma^2=0.01$','Heuristic PCS, $\sigma^2=0.01$'};
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend(legendLabels,'Interpreter','latex');
set(h_legend,'FontName','Times New Roman','FontSize',13,'FontWeight','normal','LineWidth',1,'Location','northeast');
xlabel('$c_0$','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
ylabel('$P_{\rm d}$','FontSize',16,'FontName','Times New Roman','Interpreter','latex');
xlim([1.06,1.38]);
% 原图纵轴为[0.75,0.95]；补充检测设置下使用完整范围，避免截断未核验的数据。
ylim([0,1]);
xticks(1.1:0.05:1.35);
yticks(0:0.2:1);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[0.11,0.12,0.87,0.86]);
drawnow;
print(gcf,'Fig8b_2024_TSP.png','-dpng','-r600');
print(gcf,'Fig8b_2024_TSP.pdf','-dpdf','-vector');

function symbols = drawWithUniform(x,p,u)
% 按同一组均匀随机数和各自的概率分布生成星座符号。
cdf = cumsum(p);
cdf(end) = 1;
index = ones(size(u));
for iq = 1:numel(x)-1
    index = index+(u>cdf(iq));
end
symbols = reshape(x(index),size(u));
end

function alpha = soCFARFactor(Pfa,Ntrain)
% 补充SO-CFAR阈值：按独立指数噪声标定；实际相关训练单元下的Pfa尚未校准。
k = (0:Ntrain-1).';
logCoefficient = gammaln(Ntrain+k)-gammaln(Ntrain)-gammaln(k+1);
probability = @(a) 2*sum(exp(logCoefficient+(Ntrain+k)*log(Ntrain/(2*Ntrain+a))));
upper = 1;
while probability(upper)>Pfa
    upper = 2*upper;
end
alpha = fzero(@(a) probability(a)-Pfa,[0,upper]);
end



function result = Fig8_PCS_MBA(cfg)
% 论文图8的共用概率设计：实现式(31)～(37)的MBA，并复用图7的启发式选解规则。
% GH模式是固定积分节点的数值实现；MC模式使用固定均匀混合提议分布进行重要性积分。
% 需要Communications Toolbox和Optimization Toolbox。
if nargin==0
    cfg = struct;
end
defaults = struct('c0Axis',1.075:0.025:1.375,'noiseVariances',[0.005,0.01],'integrationMode','GH','ghOrder',32,'NmcIntegral',65536,'Ncheck',50000,'maxMBA',200,'maxNewton',100,'paperTolerance',1e-5,'momentTolerance',1e-11,'gapTolerance',1e-7,'seed',42);
names = fieldnames(defaults);
for iname = 1:numel(names)
    if ~isfield(cfg,names{iname})
        cfg.(names{iname}) = defaults.(names{iname});
    end
end
assert(cfg.Ncheck>=1000,'独立AIR检查的样本数过小。');
x = qammod((0:63).',64,'UnitAveragePower',true);
power = abs(x).^2;
nC0 = numel(cfg.c0Axis);
nNoise = numel(cfg.noiseVariances);
result = struct('schema',1,'cfg',cfg,'x',x);
result.pOptimal = zeros(64,nC0,nNoise);
result.pHeuristic = zeros(64,nC0);
result.AIR = zeros(2*nNoise,nC0);
result.AIRMonteCarlo = zeros(2*nNoise,nC0);
result.AIRStandardError = zeros(2*nNoise,nC0);
result.diagnostics = cell(nNoise,nC0);
result.integrationMassError = zeros(nNoise,1);
for ic0 = 1:nC0
    result.pHeuristic(:,ic0) = heuristicProbability(x,cfg.c0Axis(ic0));
end
for inoise = 1:nNoise
    rng(cfg.seed+inoise,'twister');
    variance = cfg.noiseVariances(inoise);
    % 论文式(23)、(35)：固定积分节点，形成行归一化的离散AWGN转移概率。
    [channel,conditionalLogMean,massError] = awgnChannel(x,variance,cfg);
    result.integrationMassError(inoise) = massError;
    fprintf('sigma^2=%.4g，积分模式%s，归一化前行质量最大误差%.3e。\n',variance,cfg.integrationMode,massError);
    for ic0 = 1:nC0
        c0 = cfg.c0Axis(ic0);
        [p,info] = mbaProbability(channel,conditionalLogMean,power,c0,cfg);
        ph = result.pHeuristic(:,ic0);
        result.pOptimal(:,ic0,inoise) = p;
        result.diagnostics{inoise,ic0} = info;
        rows = [2*inoise-1,2*inoise];
        result.AIR(rows(1),ic0) = info.history(end,2);
        result.AIR(rows(2),ic0) = sum(ph.*channelDivergence(ph,channel,conditionalLogMean))/log(2);
        assert(result.AIR(rows(1),ic0)>=result.AIR(rows(2),ic0)-1e-7,'MBA的AIR低于可行启发式解。');
        % 论文式(25)、(26)：用全新的AWGN样本独立检查AIR，不复用优化积分节点。
        [result.AIRMonteCarlo(rows(1),ic0),result.AIRStandardError(rows(1),ic0)] = airMonteCarlo(x,p,variance,cfg.Ncheck);
        [result.AIRMonteCarlo(rows(2),ic0),result.AIRStandardError(rows(2),ic0)] = airMonteCarlo(x,ph,variance,cfg.Ncheck);
        discrepancy = abs(result.AIR(rows,ic0)-result.AIRMonteCarlo(rows,ic0));
        assert(all(discrepancy<=8*result.AIRStandardError(rows,ic0)+0.003),'AIR独立检查失败，请提高积分精度。');
        fprintf('c0=%.3f：MBA %.6f，启发式 %.6f，迭代%d，约束残差%.2e，最优性界%.2e bit。\n',c0,result.AIR(rows(1),ic0),result.AIR(rows(2),ic0),size(info.history,1),info.constraintResidual,info.gap);
    end
end
end

function [channel,conditionalLogMean,massError] = awgnChannel(x,variance,cfg)
% 论文式(35)：以h(y)=sum_x p(y|x)/Q为固定提议分布，积分权重包含p(y|x)/h(y)。
Q = numel(x);
if strcmpi(cfg.integrationMode,'GH')
    [nodes,weights] = hermiteRule(cfg.ghOrder);
    [u,v] = ndgrid(nodes,nodes);
    [wu,wv] = ndgrid(weights,weights);
    noiseNodes = sqrt(variance)*(u(:)+1i*v(:));
    baseWeights = wu(:).*wv(:)/pi;
    observations = bsxfun(@plus,noiseNodes,x.');
    observations = observations(:).';
    logBaseWeight = repmat(log(baseWeights)-log(Q),Q,1).';
elseif strcmpi(cfg.integrationMode,'MC')
    observations = reshape(x(randi(Q,1,cfg.NmcIntegral)),1,[]);
    observations = observations+sqrt(variance/2)*(randn(1,cfg.NmcIntegral)+1i*randn(1,cfg.NmcIntegral));
    logBaseWeight = -log(cfg.NmcIntegral)*ones(1,cfg.NmcIntegral);
else
    error('integrationMode必须为GH或MC。');
end
logKernel = -abs(bsxfun(@minus,x,observations)).^2/variance;
logProposal = logSumExp(logKernel,1)-log(Q);
logChannel = bsxfun(@plus,bsxfun(@minus,logKernel,logProposal),logBaseWeight);
rowLogMass = logSumExp(logChannel,2);
massError = max(abs(exp(rowLogMass)-1));
% 数值补充：每行归一化，使有限节点下的转移概率之和为1；massError记录离散化误差。
logChannel = bsxfun(@minus,logChannel,rowLogMass);
channel = exp(logChannel);
conditionalLogMean = sum(channel.*log(max(channel,realmin)),2);
end

function [nodes,weights] = hermiteRule(order)
% 式(35)积分的补充实现：权函数exp(-z^2)的Gauss-Hermite节点与权重。
offDiagonal = sqrt((1:order-1)/2);
[V,D] = eig(diag(offDiagonal,1)+diag(offDiagonal,-1));
[nodes,index] = sort(diag(D));
weights = sqrt(pi)*(V(1,index).^2).';
end

function [p,info] = mbaProbability(channel,conditionalLogMean,power,c0,cfg)
% 论文算法1：均匀初始化，交替更新后验q与满足C1、C2、C3的概率p。
assert(c0>457/441 && c0<143/63,'本MBA实现要求c0严格位于可达区间内部。');
p = ones(numel(power),1)/numel(power);
features = [power.^2-c0,power-1];
lambda = [];
history = zeros(cfg.maxMBA,6);
converged = false;
for iteration = 1:cfg.maxMBA
    divergence = channelDivergence(p,channel,conditionalLogMean);
    % 论文式(31)、(35)：E{log q(X|Y)|X=x}=log p(x)+D[p(Y|x)||p(Y)]。
    logPosteriorMean = log(max(p,realmin))+divergence;
    % 论文式(32)～(37)：lambda(1)对应四阶矩，lambda(2)对应平均功率。
    [pnew,lambda,newtonSteps] = projectProbability(logPosteriorMean,features,lambda,cfg);
    newDivergence = channelDivergence(pnew,channel,conditionalLogMean);
    AIR = sum(pnew.*newDivergence)/log(2);
    probabilityChange = sum((pnew-p).^2);
    residual = max(abs(features.'*pnew));
    % 补充最优性检查：由MI的支撑超平面得到容量上界，差值是KKT最优性界。
    upperBound = max(newDivergence-features*lambda)/log(2);
    gap = max(0,upperBound-AIR);
    history(iteration,:) = [iteration,AIR,probabilityChange,gap,residual,newtonSteps];
    if iteration>1
        assert(AIR>=history(iteration-1,2)-1e-9,'固定离散信道上的MBA目标发生下降。');
    end
    p = pnew;
    % 原文停止条件之外，同时要求矩约束与最优性界收敛，避免过早停止。
    if probabilityChange<=cfg.paperTolerance && residual<=cfg.momentTolerance && gap<=cfg.gapTolerance
        converged = true;
        break;
    end
end
assert(converged,'MBA达到最大迭代次数但未通过收敛检查。');
info.history = history(1:iteration,:);
info.lambda = lambda;
info.gap = gap;
info.constraintResidual = max(abs([sum(p)-1;sum(p.*power)-1;sum(p.*power.^2)-c0]));
assert(info.constraintResidual<1e-8,'MBA概率不满足约束。');
end

function divergence = channelDivergence(p,channel,conditionalLogMean)
% 论文式(25)、(31)、(35)：同时用于AIR与后验对数的条件期望。
outputProbability = p.'*channel;
divergence = conditionalLogMean-channel*log(max(outputProbability,realmin)).';
end

function [p,lambda,steps] = projectProbability(b,features,lambda,cfg)
% 论文式(33)：除以sum(g)后求同一零点；重排为四阶矩、功率两个残差。
% 论文式(36)：以粗二维网格初始化；后续迭代以上次乘子为中心。
if isempty(lambda)
    center = [0;0];
    gridAxis = -32:4:32;
else
    center = lambda;
    gridAxis = -2:0.5:2;
end
bestResidual = inf;
for i1 = 1:numel(gridAxis)
    for i2 = 1:numel(gridAxis)
        trial = center+[gridAxis(i1);gridAxis(i2)];
        logWeight = b-features*trial;
        trialP = exp(logWeight-logSumExp(logWeight,1));
        trialResidual = norm(features.'*trialP);
        if trialResidual<bestResidual
            bestResidual = trialResidual;
            lambda = trial;
        end
    end
end
for steps = 1:cfg.maxNewton
    logWeight = b-features*lambda;
    dualObjective = logSumExp(logWeight,1);
    p = exp(logWeight-dualObjective);
    residual = features.'*p;
    if max(abs(residual))<=cfg.momentTolerance
        return;
    end
    % 论文式(37)的稳定实现：归一化残差Jacobian为负的特征协方差矩阵。
    covariance = features.'*bsxfun(@times,features,p)-residual*residual.';
    assert(rcond(covariance)>1e-14,'Newton的Jacobian接近奇异。');
    direction = covariance\residual;
    stepSize = 1;
    accepted = false;
    for iline = 1:60
        trial = lambda+stepSize*direction;
        trialObjective = logSumExp(b-features*trial,1);
        if trialObjective<=dualObjective-1e-4*stepSize*(residual.'*direction)+1e-14
            lambda = trial;
            accepted = true;
            break;
        end
        stepSize = stepSize/2;
    end
    assert(accepted,'Newton回溯搜索失败。');
end
error('Newton达到最大迭代次数但矩约束未收敛。');
end

function value = logSumExp(a,dimension)
% 论文式(31)、(32)、(34)的数值稳定实现，避免指数上溢或下溢。
maximum = max(a,[],dimension);
value = maximum+log(sum(exp(bsxfun(@minus,a,maximum)),dimension));
end

function [AIR,standardError] = airMonteCarlo(x,p,variance,NMC)
% 论文式(23)、(25)、(26)：独立生成y=x+n，以信息密度均值检查AIR。
u = rand(NMC,1);
cdf = cumsum(p);
cdf(end) = 1;
index = ones(NMC,1);
for iq = 1:numel(x)-1
    index = index+(u>cdf(iq));
end
noise = sqrt(variance/2)*(randn(NMC,1)+1i*randn(NMC,1));
y = x(index)+noise;
logTerms = bsxfun(@plus,-abs(bsxfun(@minus,y,x.')).^2/variance,log(max(p,realmin)).');
informationDensity = (-abs(noise).^2/variance-logSumExp(logTerms,2))/log(2);
AIR = mean(informationDensity);
standardError = std(informationDensity)/sqrt(NMC);
end

function p = heuristicProbability(x,c0)
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