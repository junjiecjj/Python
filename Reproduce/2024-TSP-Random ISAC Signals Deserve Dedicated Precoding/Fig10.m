%% Fig10.m
% Fig. 10 reproduction:
% S&C tradeoff under different precoding schemes with N_T = N_R = 32, SNR = 16 dB.
%
% Methods:
%   1) DetOpt: Eq. (50), deterministic benchmark
%   2) DIP:    Eq. (46), penalty-based SGP-AO
%   3) DDP:    Eq. (37)-(44), penalty-based AO
%
% Important:
%   - The main simulation is intentionally NOT wrapped in a main function for easy debugging.
%   - CVX + SeDuMi is used for the complex Hermitian covariance subproblems.
%   - The paper does not explicitly provide the realization of H_c, nor the exact sequences of rho and nu(t).
%     These reproduction-specific parameters are grouped below and can be tuned if necessary.

clear;
clc;
close all;

rng(42);

% SDPT3 在当前 MATLAB R2024b + CVX 环境中存在 solver shim 兼容问题。
% 已通过最小 SDP 与 det_rootn 测试确认 SeDuMi 工作正常，因此 Fig.10 统一使用 SeDuMi。
cvx_solver sedumi;

%% Plot settings
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultAxesLabelFontSizeMultiplier',1);
set(groot,'defaultAxesTitleFontSizeMultiplier',1);
set(groot,'defaultLegendFontName','Times New Roman');
set(groot,'defaultLegendFontSize',14);
set(groot,'defaultLineLineWidth',2);
set(groot,'defaultLineMarkerSize',7);

%% Fig. 10 parameters from the paper
NT = 32;
NR = 32;
Nu = 4;

sigma2_s = 1;                                  % 0 dBm
sigma2_c = 1;                                  % 0 dBm
SNR_dB = 16;
P = sigma2_s*10^(SNR_dB/10);

N = 100;                                       % Number of Gaussian realizations
batchSize = 10;                                % Mini-batch size
tmax = 30;                                     % Table I
tau0 = 1e-3;                                   % Table I
xi0 = 0.1;                                     % Table I

LSet = [24, 32];

% Fig. 10 contains five communication-rate operating points.
% The paper does not explicitly list their R0 values. These ratios are inferred from Fig. 10.
rateRatio = [0.64, 0.76, 0.88, 0.96, 0.999];

%% Reproduction-specific AO parameters not explicitly specified in the paper
rho0 = 0.02;
rhoGrowth = 1.25;
rhoMax = 1e4;

nu0 = 1.0;
nuDecay = 0.15;
maxBacktracking = 20;

%% Generate R_H
% The paper states that the eigenvalues of R_H follow U[1,10].
% The eigenvectors / exact realization are not reported, so R_H is taken diagonal here.
lambda_H = sort(1+9*rand(NT,1));
RHinv = diag(1./lambda_H);

%% Generate H_c
% The paper does not provide the exact H_c realization.
% Use a fixed i.i.d. CSCG realization for reproducibility.
rng(4);
Hc = (randn(Nu,NT)+1j*randn(Nu,NT))/sqrt(2);

%% Communication-optimal water-filling and Rmax
WComm = communicationWaterFilling(Hc,P,sigma2_c);
Rmax = communicationRate(WComm,Hc,sigma2_c);
R0Set = rateRatio*Rmax;

fprintf('============================================================\n');
fprintf('Fig. 10 reproduction\n');
fprintf('N_T = %d, N_R = %d, N_u = %d, SNR = %.1f dB\n',NT,NR,Nu,SNR_dB);
fprintf('Rmax = %.6f bps/Hz\n',Rmax);
fprintf('R0 = ');
fprintf('%.6f  ',R0Set);
fprintf('\n============================================================\n');

%% Allocate results
result = struct();

%% Calculate L = 24 and L = 32
for iL = 1:length(LSet)

    L = LSet(iL);

    % Training Gaussian realizations for DIP
    rng(100+iL);
    STrain = (randn(NT,L,N)+1j*randn(NT,L,N))/sqrt(2);
    CTrain = complex(zeros(NT,NT,N));

    for n = 1:N
        CTrain(:,:,n) = STrain(:,:,n)*STrain(:,:,n)';
    end

    % Independent Gaussian realizations for performance evaluation and DDP
    rng(1000+iL);
    STest = (randn(NT,L,N)+1j*randn(NT,L,N))/sqrt(2);
    CTest = complex(zeros(NT,NT,N));

    for n = 1:N
        CTest(:,:,n) = STest(:,:,n)*STest(:,:,n)';
    end

    rateDetOpt = zeros(1,length(R0Set));
    rateDIP = zeros(1,length(R0Set));
    rateDDP = zeros(1,length(R0Set));

    JDetOpt = zeros(1,length(R0Set));
    JDIP = zeros(1,length(R0Set));
    JDDP = zeros(1,length(R0Set));

    meanViolationDDP = zeros(1,length(R0Set));

    %% Sweep communication-rate requirements
    for iR = 1:length(R0Set)

        R0 = R0Set(iR);

        fprintf('\n------------------------------------------------------------\n');
        fprintf('L = %d, R0 = %.6f bps/Hz\n',L,R0);
        fprintf('------------------------------------------------------------\n');

        %% DetOpt: Eq. (50)
        % Covariance-domain equivalent:
        % min tr[(R_H^{-1}+L/(sigma_s^2*N_R)*Omega)^(-1)]
        % s.t. tr(Omega)<=P, log2 det(I+H_c Omega H_c^H/sigma_c^2)>=R0, Omega>=0.
        %
        % The Schur-complement variable Z is used instead of trace_inv for robust CVX handling:
        % [Delta I; I Z] >= 0  =>  Z >= Delta^{-1}.

        cvx_begin sdp quiet
            cvx_precision high
            variable OmegaDet(NT,NT) hermitian semidefinite
            variable ZDet(NT,NT) hermitian semidefinite

            expression DeltaDet(NT,NT)
            expression RateMatrixDet(Nu,Nu)

            DeltaDet = RHinv+L/(sigma2_s*NR)*OmegaDet;
            RateMatrixDet = eye(Nu)+Hc*OmegaDet*Hc'/sigma2_c;

            minimize(real(trace(ZDet)))

            subject to
                [DeltaDet, eye(NT); eye(NT), ZDet] >= 0;
                real(trace(OmegaDet)) <= P;
                % log2 det(RateMatrixDet) >= R0
                % 等价于 det(RateMatrixDet)^(1/Nu) >= 2^(R0/Nu)
                det_rootn(RateMatrixDet) >= 2^(R0/Nu);
        cvx_end

        if ~contains(cvx_status,'Solved')
            error('DetOpt CVX failed at L=%d, R0=%.6f. Status: %s',L,R0,cvx_status);
        end

        WDetOpt = matrixSqrtPSD(OmegaDet);
        rateDetOpt(iR) = communicationRate(WDetOpt,Hc,sigma2_c);
        JDetOpt(iR) = ELMMSE(WDetOpt,CTest,RHinv,sigma2_s,NR);

        fprintf('DetOpt: Rate = %.6f, |Rate-R0| = %.6f, Normalized ELMMSE = %.6f dB\n',rateDetOpt(iR),abs(rateDetOpt(iR)-R0),10*log10(JDetOpt(iR)/(NT*NR)));

        %% DIP: Eq. (46), penalty-based SGP-AO
        W = WComm;
        rho = rho0;
        previousSensingObjective = ELMMSE(W,CTrain,RHinv,sigma2_s,NR);

        for t = 1:tmax

            Omega = solveOmegaCVX(W,Hc,sigma2_c,R0);
            batchIndex = randperm(N,batchSize);
            gradSensing = stochasticGradient(W,CTrain(:,:,batchIndex),RHinv,sigma2_s,NR);
            gradPenalty = rho*(W*W'-Omega)*W;
            grad = gradSensing+gradPenalty;

            hOld = batchObjective(W,CTrain(:,:,batchIndex),RHinv,sigma2_s,NR)+rho/2*norm(Omega-W*W','fro')^2;
            nu = nu0/(1+nuDecay*(t-1));

            accepted = false;

            for ib = 1:maxBacktracking
                WCandidate = projectPower(W-nu*grad,P);
                hNew = batchObjective(WCandidate,CTrain(:,:,batchIndex),RHinv,sigma2_s,NR)+rho/2*norm(Omega-WCandidate*WCandidate','fro')^2;

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
            currentSensingObjective = ELMMSE(W,CTrain,RHinv,sigma2_s,NR);
            xi = abs(communicationRate(W,Hc,sigma2_c)-R0);

            if abs(previousSensingObjective-currentSensingObjective) <= tau0 && xi <= xi0
                break;
            end

            previousSensingObjective = currentSensingObjective;
            rho = min(rho*rhoGrowth,rhoMax);
        end

        WDIP = W;
        rateDIP(iR) = communicationRate(WDIP,Hc,sigma2_c);
        JDIP(iR) = ELMMSE(WDIP,CTest,RHinv,sigma2_s,NR);

        fprintf('DIP:    Rate = %.6f, |Rate-R0| = %.6f, Normalized ELMMSE = %.6f dB, Iter = %d\n',rateDIP(iR),abs(rateDIP(iR)-R0),10*log10(JDIP(iR)/(NT*NR)),t);

        if abs(rateDIP(iR)-R0) > xi0
            warning('DIP has not reached the Fig.10 Pareto boundary: |Rate-R0| = %.6f > %.6f.',abs(rateDIP(iR)-R0),xi0);
        end

        %% DDP: Eq. (37)-(44), penalty-based AO
        JDDPSample = zeros(1,N);
        rateDDPSample = zeros(1,N);
        violationDDPSample = zeros(1,N);

        for n = 1:N

            W = WComm;
            rho = rho0;
            previousSensingObjective = singleLMMSE(W,CTest(:,:,n),RHinv,sigma2_s,NR);

            for t = 1:tmax

                Omega = solveOmegaCVX(W,Hc,sigma2_c,R0);
                gradSensing = stochasticGradient(W,CTest(:,:,n),RHinv,sigma2_s,NR);
                gradPenalty = rho*(W*W'-Omega)*W;
                grad = gradSensing+gradPenalty;

                hOld = singleLMMSE(W,CTest(:,:,n),RHinv,sigma2_s,NR)+rho/2*norm(Omega-W*W','fro')^2;
                nu = nu0/(1+nuDecay*(t-1));

                accepted = false;

                for ib = 1:maxBacktracking
                    WCandidate = projectPower(W-nu*grad,P);
                    hNew = singleLMMSE(WCandidate,CTest(:,:,n),RHinv,sigma2_s,NR)+rho/2*norm(Omega-WCandidate*WCandidate','fro')^2;

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
                currentSensingObjective = singleLMMSE(W,CTest(:,:,n),RHinv,sigma2_s,NR);
                xi = abs(communicationRate(W,Hc,sigma2_c)-R0);

                if abs(previousSensingObjective-currentSensingObjective) <= tau0 && xi <= xi0
                    break;
                end

                previousSensingObjective = currentSensingObjective;
                rho = min(rho*rhoGrowth,rhoMax);
            end

            JDDPSample(n) = singleLMMSE(W,CTest(:,:,n),RHinv,sigma2_s,NR);
            rateDDPSample(n) = communicationRate(W,Hc,sigma2_c);
            violationDDPSample(n) = abs(rateDDPSample(n)-R0);
        end

        JDDP(iR) = mean(JDDPSample);
        rateDDP(iR) = mean(rateDDPSample);
        meanViolationDDP(iR) = mean(violationDDPSample);

        fprintf('DDP:    Rate = %.6f, Mean|Rate-R0| = %.6f, Normalized ELMMSE = %.6f dB\n',rateDDP(iR),meanViolationDDP(iR),10*log10(JDDP(iR)/(NT*NR)));

        if meanViolationDDP(iR) > xi0
            warning('DDP has not reached the Fig.10 Pareto boundary: Mean|Rate-R0| = %.6f > %.6f.',meanViolationDDP(iR),xi0);
        end
    end

    %% Store results
    fieldName = sprintf('L%d',L);

    result.(fieldName).DetOptRate = rateDetOpt;
    result.(fieldName).DIPRate = rateDIP;
    result.(fieldName).DDPRate = rateDDP;

    result.(fieldName).DetOpt = 10*log10(JDetOpt/(NT*NR));
    result.(fieldName).DIP = 10*log10(JDIP/(NT*NR));
    result.(fieldName).DDP = 10*log10(JDDP/(NT*NR));

    result.(fieldName).DDPMeanViolation = meanViolationDDP;
end

%% Print final rate arrays for debugging
fprintf('\n============================================================\n');
fprintf('Final communication-rate arrays\n');
fprintf('============================================================\n');

fprintf('L=24 DetOpt Rate: ');
fprintf('%.6f  ',result.L24.DetOptRate);
fprintf('\n');

fprintf('L=24 DIP Rate:    ');
fprintf('%.6f  ',result.L24.DIPRate);
fprintf('\n');

fprintf('L=24 DDP Rate:    ');
fprintf('%.6f  ',result.L24.DDPRate);
fprintf('\n');

fprintf('L=32 DetOpt Rate: ');
fprintf('%.6f  ',result.L32.DetOptRate);
fprintf('\n');

fprintf('L=32 DIP Rate:    ');
fprintf('%.6f  ',result.L32.DIPRate);
fprintf('\n');

fprintf('L=32 DDP Rate:    ');
fprintf('%.6f  ',result.L32.DDPRate);
fprintf('\n');

%% Plot Fig. 10
figure('Color','w','Position',[100,100,800,600]);
hold on;

colorDetOpt = '#F65314';
colorDIP = '#00A1F1';
colorDDP = '#8A2BE2';

% L = 24: solid lines and hollow markers
plot(result.L24.DetOptRate,result.L24.DetOpt,'Color',colorDetOpt,'LineStyle','-','LineWidth',2,'Marker','>','MarkerSize',8,'MarkerFaceColor','w','DisplayName','DetOpt, L = 24');
plot(result.L24.DIPRate,result.L24.DIP,'Color',colorDIP,'LineStyle','-','LineWidth',2,'Marker','o','MarkerSize',7,'MarkerFaceColor','w','DisplayName','DIP, L = 24');
plot(result.L24.DDPRate,result.L24.DDP,'Color',colorDDP,'LineStyle','-','LineWidth',2,'Marker','s','MarkerSize',7,'MarkerFaceColor','w','DisplayName','DDP, L = 24');

% L = 32: dashed lines and filled markers
plot(result.L32.DetOptRate,result.L32.DetOpt,'Color',colorDetOpt,'LineStyle','--','LineWidth',2,'Marker','>','MarkerSize',8,'MarkerFaceColor',colorDetOpt,'DisplayName','DetOpt, L = 32');
plot(result.L32.DIPRate,result.L32.DIP,'Color',colorDIP,'LineStyle','--','LineWidth',2,'Marker','o','MarkerSize',7,'MarkerFaceColor',colorDIP,'DisplayName','DIP, L = 32');
plot(result.L32.DDPRate,result.L32.DDP,'Color',colorDDP,'LineStyle','--','LineWidth',2,'Marker','s','MarkerSize',7,'MarkerFaceColor',colorDDP,'DisplayName','DDP, L = 32');

legend1 = legend('Location','northwest','NumColumns',2,'Box','on');
set(legend1,'FontName','Times New Roman','FontSize',14,'Color','none');

ax = gca;
bw = 2;
ax.LineWidth = bw;
ax.FontName = 'Times New Roman';
ax.FontSize = 18;
ax.TickDir = 'in';
ax.XAxis.TickDirection = 'in';
ax.YAxis.TickDirection = 'in';
ax.Box = 'on';
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridLineStyle = '--';
ax.GridAlpha = 0.25;

xlabel('Communication Rate [bps/Hz]','FontName','Times New Roman','FontSize',18);
ylabel('Normalized ELMMSE [dB]','FontName','Times New Roman','FontSize',18);

xlim([20.5,33]);
ylim([-16,-10]);
xticks(22:2:32);
yticks(-16:1:-10);

set(gcf,'PaperPositionMode','auto');
print(gcf,'Fig10_TSP_MATLAB','-dpdf','-bestfit');
% print(gcf,'Fig10_TSP_MATLAB','-dpng','-r300');

hold off;


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
    Nu = size(Hc,1);
    A = eye(Nu)+Hc*W*W'*Hc'/sigma2_c;
    A = (A+A')/2;
    eigA = real(eig(A));
    eigA(eigA<eps) = eps;
    R = sum(log2(eigA));
end


function W = communicationWaterFilling(Hc,P,sigma2_c)
    [~,S,V] = svd(Hc,'econ');
    singularValue = diag(S);
    gain = singularValue.^2/sigma2_c;

    muLeft = 0;
    muRight = 1;

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


function J = singleLMMSE(W,C,RHinv,sigma2_s,NR)
    Delta = RHinv+1/(sigma2_s*NR)*W*C*W';
    Delta = (Delta+Delta')/2;
    J = real(trace(Delta\eye(size(Delta))));
end


function J = batchObjective(W,CSet,RHinv,sigma2_s,NR)
    numSamples = size(CSet,3);
    J = 0;

    for n = 1:numSamples
        J = J+singleLMMSE(W,CSet(:,:,n),RHinv,sigma2_s,NR);
    end

    J = J/numSamples;
end


function J = ELMMSE(W,CSet,RHinv,sigma2_s,NR)
    J = batchObjective(W,CSet,RHinv,sigma2_s,NR);
end


function grad = stochasticGradient(W,CSet,RHinv,sigma2_s,NR)
    numSamples = size(CSet,3);
    NT = size(W,1);
    c = 1/(sigma2_s*NR);
    grad = complex(zeros(NT,NT));

    for n = 1:numSamples
        C = CSet(:,:,n);
        Delta = RHinv+c*W*C*W';
        Delta = (Delta+Delta')/2;
        WC = W*C;
        grad = grad-c*(Delta\(Delta\WC));
    end

    grad = grad/numSamples;
end


function Omega = solveOmegaCVX(W,Hc,sigma2_c,R0)
    NT = size(W,1);
    Nu = size(Hc,1);
    A = W*W';
    currentRate = communicationRate(W,Hc,sigma2_c);

    % If A itself is feasible, Eq. (40) has the exact solution Omega=A,
    % since the minimum possible Frobenius distance is zero.
    if currentRate >= R0-1e-10
        Omega = A;
        return;
    end

    cvx_begin sdp quiet
        cvx_precision high
        variable OmegaVar(NT,NT) hermitian semidefinite
        expression RateMatrix(Nu,Nu)

        RateMatrix = eye(Nu)+Hc*OmegaVar*Hc'/sigma2_c;

        minimize(norm(OmegaVar-A,'fro'))

        subject to
            % log2 det(RateMatrix) >= R0
            % 等价于 det(RateMatrix)^(1/Nu) >= 2^(R0/Nu)
            det_rootn(RateMatrix) >= 2^(R0/Nu);
    cvx_end

    if ~contains(cvx_status,'Solved')
        error('Eq. (40) CVX failed. Status: %s',cvx_status);
    end

    Omega = (OmegaVar+OmegaVar')/2;
end
