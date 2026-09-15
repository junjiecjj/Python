%% Fig14_JSAC.m
% 复现：2025 JSAC tutorial Fig.14
% Sensing With Communication Signals: From Information Theory to Signal Processing
% Fig.14: S&C performance tradeoff under Baseline / DIP / DDP precoding.
%
% 论文明确参数：N_t=N_s=32，N=20 和 32，SNR=15 dB。
% 其算法来自同团队 2024 TSP [70]：Random ISAC Signals Deserve Dedicated Precoding。
% Baseline：确定性 LMMSE 优化；DIP：Penalty-Based SGP-AO；DDP：Penalty-Based AO。
%
% 注意：论文没有公开 H_c 的具体 realization，也没有公开 rho 和 nu(t) 的精确更新序列。
% 因此下面固定 H_c~CN(0,1) 的 seed，并将这些复现参数集中放在参数区，便于调试。
% Fig.14 中约有 9 个 rate operating points，这里按图反推 R0/Rmax=0.60:0.05:1.00，
% 最后一点用 0.999 代替 1.00 以避免容量边界上的数值不可行。

clear all; clc; close all;
rng(42);

% 修复 MATLAB R2024b 下 CVX 数值 vec.m 未自动加入路径的问题
cvxRoot = fileparts(which('cvx_setup'));
vecPath = fullfile(cvxRoot,'functions','vec_');

if exist(vecPath,'dir')
    addpath(vecPath,'-begin');
    clear vec;
    rehash path;
end

% 上一次 CVX 若因报错中断，可能残留“正在构造模型”的状态。
% clear all 会同时清除该残留状态；随后再选择 SeDuMi。
cvx_clear;
cvx_solver sedumi;

%% 绘图设置
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultAxesLabelFontSizeMultiplier',1);
set(groot,'defaultAxesTitleFontSizeMultiplier',1);
set(groot,'defaultLegendFontName','Times New Roman');
set(groot,'defaultLegendFontSize',14);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',7);

%% Fig.14 参数
NT = 32;
Ns = 32;
Nc = 4;                                        % 继承 [70] Table I
sigma2_s = 1;
sigma2_c = 1;
SNR_dB = 15;
PT = sigma2_s*10^(SNR_dB/10);
NumMC = 100;                                   % 继承 [70] Table I 中 Gaussian realizations 数量
batchSize = 10;                                % [70] Table I / simulation setup
tmax = 30;
tau0 = 1e-3;
xi0 = 0.1;
NSet = [20, 32];
rateRatio = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.999];

%% 论文未明确给出的 AO 数值参数
rho0 = 0.02;
rhoGrowth = 1.25;
rhoMax = 1e4;
nu0 = 1.0;
nuDecay = 0.15;
maxBacktracking = 20;

%% 产生 R_H
% [70] 说明 R_H 的特征值服从 U[1,10]；未给出具体 realization/eigenvectors。
lambda_H = sort(1+9*rand(NT,1));
RHinv = diag(1./lambda_H);

%% 产生通信信道 H_c
% 论文未给出 H_c realization。固定 seed=4 的 i.i.d. CSCG realization；
% 在 SNR=15 dB 下该 realization 的 Rmax 约位于 Fig.14 横轴末端附近。
rng(4);
Hc = (randn(Nc,NT)+1j*randn(Nc,NT))/sqrt(2);

%% 通信最优 water-filling 及 Rmax
WComm = communicationWaterFilling(Hc,PT,sigma2_c);
Rmax = communicationRate(WComm,Hc,sigma2_c);
R0Set = rateRatio*Rmax;

fprintf('============================================================\n');
fprintf('Fig.14 reproduction\n');
fprintf('N_t = %d, N_s = %d, N_c = %d, SNR = %.1f dB\n',NT,Ns,Nc,SNR_dB);
fprintf('Rmax = %.6f bps/Hz\n',Rmax);
fprintf('R0 = '); fprintf('%.6f  ',R0Set); fprintf('\n');
fprintf('============================================================\n');

%% 结果
result = struct();

%% 分别计算 N=20 和 N=32
for iN = 1:length(NSet)
    N = NSet(iN);

    % DIP 离线优化使用 Gaussian training realizations
    rng(100+iN);
    STrain = (randn(NT,N,NumMC)+1j*randn(NT,N,NumMC))/sqrt(2);
    CTrain = complex(zeros(NT,NT,NumMC));
    for m = 1:NumMC
        CTrain(:,:,m) = STrain(:,:,m)*STrain(:,:,m)';
    end

    % Baseline、DIP 的性能评价以及 DDP 实时设计使用独立 test realizations
    rng(1000+iN);
    STest = (randn(NT,N,NumMC)+1j*randn(NT,N,NumMC))/sqrt(2);
    CTest = complex(zeros(NT,NT,NumMC));
    for m = 1:NumMC
        CTest(:,:,m) = STest(:,:,m)*STest(:,:,m)';
    end

    rateBaseline = zeros(1,length(R0Set));
    rateDIP = zeros(1,length(R0Set));
    rateDDP = zeros(1,length(R0Set));
    JBaseline = zeros(1,length(R0Set));
    JDIP = zeros(1,length(R0Set));
    JDDP = zeros(1,length(R0Set));
    meanViolationDDP = zeros(1,length(R0Set));

    %% 扫描通信速率约束
    for iR = 1:length(R0Set)
        R0 = R0Set(iR);
        fprintf('\n------------------------------------------------------------\n');
        fprintf('N = %d, R0 = %.6f bps/Hz\n',N,R0);
        fprintf('------------------------------------------------------------\n');

        %% Baseline：deterministic LMMSE + rate/power constraints
        % min tr[(R_H^{-1}+N/(sigma_s^2*N_s)*Omega)^(-1)]
        % s.t. tr(Omega)<=P_T, log2 det(I+H_c Omega H_c^H/sigma_c^2)>=R0, Omega>=0.
        % 用 Schur complement 代替 trace_inv： [Delta I; I Z]>=0 => Z>=Delta^{-1}。
        cvx_begin sdp quiet
            cvx_precision high
            variable OmegaBase(NT,NT) hermitian semidefinite
            variable ZBase(NT,NT) hermitian semidefinite
            expression DeltaBase(NT,NT)
            expression RateMatrixBase(Nc,Nc)
            DeltaBase = RHinv+N/(sigma2_s*Ns)*OmegaBase;
            RateMatrixBase = eye(Nc)+Hc*OmegaBase*Hc'/sigma2_c;
            minimize(real(trace(ZBase)))
            subject to
                [DeltaBase, eye(NT); eye(NT), ZBase] >= 0;
                real(trace(OmegaBase)) <= PT;
                det_rootn(RateMatrixBase) >= 2^(R0/Nc);
        cvx_end
        if ~contains(cvx_status,'Solved')
            error('Baseline CVX failed at N=%d, R0=%.6f. Status: %s',N,R0,cvx_status);
        end
        WBaseline = matrixSqrtPSD(OmegaBase);
        rateBaseline(iR) = communicationRate(WBaseline,Hc,sigma2_c);
        JBaseline(iR) = ELMMSE(WBaseline,CTest,RHinv,sigma2_s,Ns);
        fprintf('Baseline: Rate = %.6f, |Rate-R0| = %.6f, Normalized ELMMSE = %.6f dB\n',rateBaseline(iR),abs(rateBaseline(iR)-R0),10*log10(JBaseline(iR)/(NT*Ns)));

        %% DIP：Eq.(101)，Penalty-Based SGP-AO
        W = WComm;
        rho = rho0;
        previousSensingObjective = ELMMSE(W,CTrain,RHinv,sigma2_s,Ns);
        for t = 1:tmax
            Omega = solveOmegaCVX(W,Hc,sigma2_c,R0);
            batchIndex = randperm(NumMC,batchSize);
            gradSensing = stochasticGradient(W,CTrain(:,:,batchIndex),RHinv,sigma2_s,Ns);
            gradPenalty = rho*(W*W'-Omega)*W;
            grad = gradSensing+gradPenalty;
            hOld = batchObjective(W,CTrain(:,:,batchIndex),RHinv,sigma2_s,Ns)+rho/2*norm(Omega-W*W','fro')^2;
            nu = nu0/(1+nuDecay*(t-1));
            accepted = false;
            for ib = 1:maxBacktracking
                WCandidate = projectPower(W-nu*grad,PT);
                hNew = batchObjective(WCandidate,CTrain(:,:,batchIndex),RHinv,sigma2_s,Ns)+rho/2*norm(Omega-WCandidate*WCandidate','fro')^2;
                if hNew <= hOld+1e-12
                    accepted = true;
                    break;
                end
                nu = nu/2;
            end
            if ~accepted
                WCandidate = W;
            end
            W = WCandidate;
            currentSensingObjective = ELMMSE(W,CTrain,RHinv,sigma2_s,Ns);
            xi = max(R0-communicationRate(W,Hc,sigma2_c),0);
            if abs(previousSensingObjective-currentSensingObjective) <= tau0 && xi <= xi0
                break;
            end
            previousSensingObjective = currentSensingObjective;
            rho = min(rho*rhoGrowth,rhoMax);
        end
        WDIP = W;
        rateDIP(iR) = communicationRate(WDIP,Hc,sigma2_c);
        JDIP(iR) = ELMMSE(WDIP,CTest,RHinv,sigma2_s,Ns);
        fprintf('DIP:      Rate = %.6f, RateViolation = %.6f, Normalized ELMMSE = %.6f dB, Iter = %d\n',rateDIP(iR),max(R0-rateDIP(iR),0),10*log10(JDIP(iR)/(NT*Ns)),t);
        if max(R0-rateDIP(iR),0) > xi0
            warning('DIP rate constraint is not satisfied: violation = %.6f > %.6f.',max(R0-rateDIP(iR),0),xi0);
        end

        %% DDP：Eq.(99)-(100)，每个数据 realization 单独执行 Penalty-Based AO
        JDDPSample = zeros(1,NumMC);
        rateDDPSample = zeros(1,NumMC);
        violationDDPSample = zeros(1,NumMC);
        for m = 1:NumMC
            W = WComm;
            rho = rho0;
            previousSensingObjective = singleLMMSE(W,CTest(:,:,m),RHinv,sigma2_s,Ns);
            for t = 1:tmax
                Omega = solveOmegaCVX(W,Hc,sigma2_c,R0);
                gradSensing = stochasticGradient(W,CTest(:,:,m),RHinv,sigma2_s,Ns);
                gradPenalty = rho*(W*W'-Omega)*W;
                grad = gradSensing+gradPenalty;
                hOld = singleLMMSE(W,CTest(:,:,m),RHinv,sigma2_s,Ns)+rho/2*norm(Omega-W*W','fro')^2;
                nu = nu0/(1+nuDecay*(t-1));
                accepted = false;
                for ib = 1:maxBacktracking
                    WCandidate = projectPower(W-nu*grad,PT);
                    hNew = singleLMMSE(WCandidate,CTest(:,:,m),RHinv,sigma2_s,Ns)+rho/2*norm(Omega-WCandidate*WCandidate','fro')^2;
                    if hNew <= hOld+1e-12
                        accepted = true;
                        break;
                    end
                    nu = nu/2;
                end
                if ~accepted
                    WCandidate = W;
                end
                W = WCandidate;
                currentSensingObjective = singleLMMSE(W,CTest(:,:,m),RHinv,sigma2_s,Ns);
                xi = max(R0-communicationRate(W,Hc,sigma2_c),0);
                if abs(previousSensingObjective-currentSensingObjective) <= tau0 && xi <= xi0
                    break;
                end
                previousSensingObjective = currentSensingObjective;
                rho = min(rho*rhoGrowth,rhoMax);
            end
            JDDPSample(m) = singleLMMSE(W,CTest(:,:,m),RHinv,sigma2_s,Ns);
            rateDDPSample(m) = communicationRate(W,Hc,sigma2_c);
            violationDDPSample(m) = max(R0-rateDDPSample(m),0);
        end
        JDDP(iR) = mean(JDDPSample);
        rateDDP(iR) = mean(rateDDPSample);
        meanViolationDDP(iR) = mean(violationDDPSample);
        fprintf('DDP:      Rate = %.6f, MeanRateViolation = %.6f, Normalized ELMMSE = %.6f dB\n',rateDDP(iR),meanViolationDDP(iR),10*log10(JDDP(iR)/(NT*Ns)));
        if meanViolationDDP(iR) > xi0
            warning('DDP average rate constraint violation = %.6f > %.6f.',meanViolationDDP(iR),xi0);
        end
    end

    fieldName = sprintf('N%d',N);
    result.(fieldName).BaselineRate = rateBaseline;
    result.(fieldName).DIPRate = rateDIP;
    result.(fieldName).DDPRate = rateDDP;
    result.(fieldName).Baseline = 10*log10(JBaseline/(NT*Ns));
    result.(fieldName).DIP = 10*log10(JDIP/(NT*Ns));
    result.(fieldName).DDP = 10*log10(JDDP/(NT*Ns));
    result.(fieldName).DDPMeanViolation = meanViolationDDP;
end

%% 打印最终结果，便于逐点排查
fprintf('\n============================================================\n');
fprintf('Final communication-rate arrays\n');
fprintf('============================================================\n');
fprintf('N=20 Baseline Rate: '); fprintf('%.6f  ',result.N20.BaselineRate); fprintf('\n');
fprintf('N=20 DIP Rate:      '); fprintf('%.6f  ',result.N20.DIPRate); fprintf('\n');
fprintf('N=20 DDP Rate:      '); fprintf('%.6f  ',result.N20.DDPRate); fprintf('\n');
fprintf('N=32 Baseline Rate: '); fprintf('%.6f  ',result.N32.BaselineRate); fprintf('\n');
fprintf('N=32 DIP Rate:      '); fprintf('%.6f  ',result.N32.DIPRate); fprintf('\n');
fprintf('N=32 DDP Rate:      '); fprintf('%.6f  ',result.N32.DDPRate); fprintf('\n');

%% 绘制 Fig.14
%%===========================================

width = 6;%设置图宽，这个不用改
height = 4;%设置图高，这个不用改
fontsize = 14;%设置图中字体大小
linewidth = 2;%设置线宽
markersize = 10;%标记大小

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

figure(1);
set(gcf, 'Units', 'inches');
set(gcf, 'Color', 'white');
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
set(gcf, 'PaperPositionMode', 'manual');

colorBaseline = '#77AC30';
colorDIP = '#00A1F1';
colorDDP = '#F65314';

% N=20：实线；N=32：虚线。Baseline 用圆圈，DIP 不加 marker，DDP 用方块。
p1 = plot(result.N20.BaselineRate, result.N20.Baseline, '-', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize', markersize, 'MarkerFaceColor', 'w'); hold on;
p1.Color = colorBaseline;

p2 = plot(result.N20.DIPRate, result.N20.DIP, '-', 'LineWidth', linewidth); hold on;
p2.Color = colorDIP;

p3 = plot(result.N20.DDPRate, result.N20.DDP, '-', 'LineWidth', linewidth, 'Marker', 's', 'MarkerSize', markersize, 'MarkerFaceColor', 'w'); hold on;
p3.Color = colorDDP;

p4 = plot(result.N32.BaselineRate, result.N32.Baseline, '--', 'LineWidth', linewidth, 'Marker', 'o', 'MarkerSize', markersize, 'MarkerFaceColor', 'w'); hold on;
p4.Color = colorBaseline;

p5 = plot(result.N32.DIPRate, result.N32.DIP, '--', 'LineWidth', linewidth); hold on;
p5.Color = colorDIP;

p6 = plot(result.N32.DDPRate, result.N32.DDP, '--', 'LineWidth', linewidth, 'Marker', 's', 'MarkerSize', markersize, 'MarkerFaceColor', 'w'); hold on;
p6.Color = colorDDP;

%-------------------------------------------------------------------
% 坐标轴、图例、标注设置
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman');

h_legend = legend('Baseline, $N=20$', 'DIP Scheme, $N=20$', 'DDP Scheme, $N=20$', ...
                  'Baseline, $N=32$', 'DIP Scheme, $N=32$', 'DDP Scheme, $N=32$', ...
                  'Interpreter', 'latex');

legendsize = 12;
set(h_legend, 'FontName', 'Times New Roman', 'FontSize', legendsize, 'FontWeight', 'normal', ...
              'LineWidth', 1, 'Location', 'Best', 'NumColumns', 2);
h_legend.Color = 'none';

labelsize = 18;
xlabel('Communication Rate [bps/s/Hz]', 'FontSize', labelsize, 'FontName', 'Times New Roman');
ylabel('Normalized ELMMSE [dB]', 'FontSize', labelsize, 'FontName', 'Times New Roman');

xlim([18 32]);
ylim([-15 -9]);
xticks(18:2:32);
yticks(-15:1:-9);

%----- Grid 设置----------------
grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.2, 'LineWidth', 1, 'GridLineWidth', 0.5, 'Layer', 'bottom');

%--------- savefig-------------
set(gca, 'Units', 'normalized');
set(gca, 'Position', [0.11, 0.13, 0.87, 0.86]);

print(gcf, 'Fig14_JSAC_MATLAB.pdf', '-dpdf', '-vector');

%% ========================================================================
% Local functions
% ========================================================================
function Wproj = projectPower(W,P)
    power = norm(W,'fro')^2;
    if power <= P
        Wproj = W;
    else
        Wproj = W*sqrt(P/power);
    end
end

function W = matrixSqrtPSD(A)
    A = (A+A')/2;
    [U,D] = eig(A);
    d = real(diag(D));
    d(d<0) = 0;
    W = U*diag(sqrt(d))*U';
end

function R = communicationRate(W,Hc,sigma2_c)
    Nc = size(Hc,1);
    A = eye(Nc)+Hc*W*W'*Hc'/sigma2_c;
    A = (A+A')/2;
    eigA = real(eig(A));
    eigA(eigA<eps) = eps;
    R = sum(log2(eigA));
end

function W = communicationWaterFilling(Hc,P,sigma2_c)
    [~,S,V] = svd(Hc,'econ');
    singularValue = diag(S);
    gain = singularValue.^2/sigma2_c;
    muLeft = 0; muRight = 1;
    while sum(max(muRight-1./gain,0)) < P
        muRight = 2*muRight;
    end
    for k = 1:100
        mu = (muLeft+muRight)/2;
        powerAllocation = max(mu-1./gain,0);
        if sum(powerAllocation) < P
            muLeft = mu;
        else
            muRight = mu;
        end
    end
    mu = (muLeft+muRight)/2;
    powerAllocation = max(mu-1./gain,0);
    Omega = V*diag(powerAllocation)*V';
    W = matrixSqrtPSD(Omega);
end

function J = singleLMMSE(W,C,RHinv,sigma2_s,Ns)
    Delta = RHinv+1/(sigma2_s*Ns)*W*C*W';
    Delta = (Delta+Delta')/2;
    J = real(trace(Delta\eye(size(Delta))));
end

function J = batchObjective(W,CSet,RHinv,sigma2_s,Ns)
    numSamples = size(CSet,3);
    J = 0;
    for m = 1:numSamples
        J = J+singleLMMSE(W,CSet(:,:,m),RHinv,sigma2_s,Ns);
    end
    J = J/numSamples;
end

function J = ELMMSE(W,CSet,RHinv,sigma2_s,Ns)
    J = batchObjective(W,CSet,RHinv,sigma2_s,Ns);
end

function grad = stochasticGradient(W,CSet,RHinv,sigma2_s,Ns)
    numSamples = size(CSet,3);
    NT = size(W,1);
    c = 1/(sigma2_s*Ns);
    grad = complex(zeros(NT,NT));
    for m = 1:numSamples
        C = CSet(:,:,m);
        Delta = RHinv+c*W*C*W';
        Delta = (Delta+Delta')/2;
        WC = W*C;
        grad = grad-c*(Delta\(Delta\WC));
    end
    grad = grad/numSamples;
end

function Omega = solveOmegaCVX(W,Hc,sigma2_c,R0)
    NT = size(W,1);
    Nc = size(Hc,1);
    A = W*W';
    currentRate = communicationRate(W,Hc,sigma2_c);
    if currentRate >= R0-1e-10
        Omega = A;
        return;
    end
    cvx_begin sdp quiet
        cvx_precision high
        variable OmegaVar(NT,NT) hermitian semidefinite
        expression RateMatrix(Nc,Nc)
        RateMatrix = eye(Nc)+Hc*OmegaVar*Hc'/sigma2_c;
        minimize(norm(OmegaVar-A,'fro'))
        subject to
            det_rootn(RateMatrix) >= 2^(R0/Nc);
    cvx_end
    if ~contains(cvx_status,'Solved')
        error('Omega subproblem CVX failed. Status: %s',cvx_status);
    end
    Omega = (OmegaVar+OmegaVar')/2;
end
