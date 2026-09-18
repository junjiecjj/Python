% Sensing With Communication Signals: From Information Theory to Signal Processing
% Fig.13: sensing-only precoding performance under Gaussian signaling
%
% Methods:
%   1) Water-Filling: JSAC Eq.(85), equivalent to TSP Eq.(8)
%   2) DIP (SGP):    JSAC Eq.(91)-(94), equivalent to TSP Eq.(22)-(26)
%   3) DDP:          JSAC Eq.(90), equivalent to TSP Theorem 1 / Eq.(19)
%
% Fig.13 parameters:
%   Case 1: N_t=N_s=32, N=24
%   Case 2: N_t=N_s=64, N=48
%   SNR = P/sigma_s^2 = 0:5:50 dB; implementation fixes P=1 and varies sigma_s^2
%   Monte-Carlo averaging: 1000 Gaussian signal realizations
%
% Parameters inherited from the cited 2024 TSP paper [70]:
%   training set size = 100
%   mini-batch size = 10
%   r_max = 1000
%   eta(r) = 10/(10+r)
%   eigenvalues of R_H ~ U[1,10]
%
% Notes:
%   - The paper does not publish the exact realization/seed of R_H.
%   - Algorithm 1 does not specify W^(1). Here the water-filling solution at
%     the same SNR is used as a deterministic initialization for DIP.
%   - R_H is taken diagonal because Gaussian S is spatially isotropic and
%     only its eigenvalues affect the averaged performance.
%   - The main simulation is intentionally NOT wrapped in a main function.

clear all; clc; close all;
rng(42);

%% 参数设置
P = 1;                                          % 固定总发射功率
SNRdB = 0:5:50;
NTrain = 100;
NMC = 1000;
batchSize = 10;
rmax = 1000;
caseSet = [32,24; 64,48];

result = struct();

%% 两组参数分别计算
for iCase = 1:size(caseSet,1)
    NT = caseSet(iCase,1);
    NR = NT;
    N = caseSet(iCase,2);

    % R_H 的特征值服从 U[1,10]，按升序排列以对应 DDP Theorem 1
    rng(42+iCase);
    lambda_H = sort(1+9*rand(NT,1),'ascend');
    Q = eye(NT);
    RHinv = diag(1./lambda_H);

    % DIP 训练集：论文 Table I 中 N=100，mini-batch size=10
    rng(100+iCase);
    STrain = (randn(NT,N,NTrain)+1j*randn(NT,N,NTrain))/sqrt(2);
    CTrain = complex(zeros(NT,NT,NTrain));
    for nTrain = 1:NTrain
        CTrain(:,:,nTrain) = STrain(:,:,nTrain)*STrain(:,:,nTrain)';
    end

    % 独立 Monte-Carlo 测试集，Fig.13 正文明确平均 1000 个 realization
    rng(1000+iCase);
    STest = (randn(NT,N,NMC)+1j*randn(NT,N,NMC))/sqrt(2);

    % 先为所有 SNR 求 Water-Filling 和 DIP，随后统一做 Monte-Carlo
    W_WF_all = complex(zeros(NT,NT,length(SNRdB)));
    W_DIP_all = complex(zeros(NT,NT,length(SNRdB)));

    for iSNR = 1:length(SNRdB)
        % 论文定义 transmit SNR = P/sigma_s^2。
        % 为保持 SGP 步长 eta(r)=10/(10+r) 与论文数值尺度一致，
        % 固定 P=1，通过改变 sigma_s^2 扫描 SNR。
        sigma2_s = P/10^(SNRdB(iSNR)/10);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Water-Filling: JSAC Eq.(85)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_WF = waterFillingPrecoder(lambda_H,Q,N,NR,sigma2_s,P);
        W_WF_all(:,:,iSNR) = W_WF;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DIP (SGP): JSAC Eq.(91)-(94)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        W_DIP = W_WF;
        rng(2000+100*iCase+iSNR);

        for r = 1:rmax
            batchIndex = randperm(NTrain,batchSize);
            grad = stochasticGradient(W_DIP,CTrain(:,:,batchIndex),RHinv,sigma2_s,NR);
            eta = 10/(10+r);
            W_DIP = projectPower(W_DIP-eta*grad,P);
        end

        W_DIP_all(:,:,iSNR) = W_DIP;
        fprintf('Case %d/%d: N_t=N_s=%d, N=%d, SNR=%2d dB, ||W_WF||_F^2=%.6f, ||W_DIP||_F^2=%.6f\n', iCase,size(caseSet,1),NT,N,SNRdB(iSNR),norm(W_WF,'fro')^2,norm(W_DIP,'fro')^2);
    end

    % Monte-Carlo 累加
    J_WF = zeros(1,length(SNRdB));
    J_DIP = zeros(1,length(SNRdB));
    J_DDP = zeros(1,length(SNRdB));

    for nMC = 1:NMC

        S = STest(:,:,nMC);

        % DDP 的 SVD 与 SNR 无关，因此每个 realization 只做一次
        [US,SigmaS,~] = svd(S);
        singularPower = real(diag(SigmaS*SigmaS'));

        for iSNR = 1:length(SNRdB)

            sigma2_s = P/10^(SNRdB(iSNR)/10);
            W_WF = W_WF_all(:,:,iSNR);
            W_DIP = W_DIP_all(:,:,iSNR);

            % Water-Filling / DIP 在同一个随机 realization 上评价
            J_WF(iSNR) = J_WF(iSNR)+singleLMMSE(W_WF,S,RHinv,sigma2_s,NR);
            J_DIP(iSNR) = J_DIP(iSNR)+singleLMMSE(W_DIP,S,RHinv,sigma2_s,NR);

            % DDP: 每一个 S realization 都单独设计 W(S)
            W_DDP = ddpPrecoderFromSVD(US,singularPower,lambda_H,Q,P,sigma2_s,NR);
            J_DDP(iSNR) = J_DDP(iSNR)+singleLMMSE(W_DDP,S,RHinv,sigma2_s,NR);
        end

        if mod(nMC,100) == 0
            fprintf('N_t=%d, N=%d: Monte-Carlo %d/%d\n',NT,N,nMC,NMC);
        end
    end

    J_WF = J_WF/NMC;
    J_DIP = J_DIP/NMC;
    J_DDP = J_DDP/NMC;

    % 与 Fig.13 的 Normalized ELMMSE 标度一致
    J_WF_dB = 10*log10(J_WF/(NT*NR));
    J_DIP_dB = 10*log10(J_DIP/(NT*NR));
    J_DDP_dB = 10*log10(J_DDP/(NT*NR));

    fieldName = sprintf('Nt%d_N%d',NT,N);
    result.(fieldName).WaterFilling = J_WF_dB;
    result.(fieldName).DIP = J_DIP_dB;
    result.(fieldName).DDP = J_DDP_dB;

    fprintf('\nN_t=N_s=%d, N=%d\n',NT,N);
    fprintf('Water-Filling [dB]: '); fprintf('%.4f  ',J_WF_dB); fprintf('\n');
    fprintf('DIP (SGP)    [dB]: '); fprintf('%.4f  ',J_DIP_dB); fprintf('\n');
    fprintf('DDP          [dB]: '); fprintf('%.4f  ',J_DDP_dB); fprintf('\n\n');
end

%%===========================================

width = 6;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 8;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

%%========================================================================================
%         绘制 Fig.13
%%========================================================================================

figure(1);
set(gcf, 'Units', 'inches');
set(gcf, 'Color', 'white');
set(gcf, 'Renderer', 'painters');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0, 0, width, height]);
set(gcf, 'PaperSize', [width, height]);
set(gcf, 'PaperPositionMode', 'manual');

colorWF = '#77AC30';
colorDIP = '#00A1F1';
colorDDP = '#F65314';

% N_t=32, N=24：实线
p1 = plot(SNRdB,result.Nt32_N24.WaterFilling,'-','LineWidth',linewidth,'Marker','o','MarkerSize',markersize,'MarkerFaceColor','w'); hold on;
p1.Color = colorWF;
p2 = plot(SNRdB,result.Nt32_N24.DIP,'-','LineWidth',linewidth);
p2.Color = colorDIP;
p3 = plot(SNRdB,result.Nt32_N24.DDP,'-','LineWidth',linewidth,'Marker','s','MarkerSize',markersize,'MarkerFaceColor','w');
p3.Color = colorDDP;

% N_t=64, N=48：虚线
p4 = plot(SNRdB,result.Nt64_N48.WaterFilling,'--','LineWidth',linewidth,'Marker','o','MarkerSize',markersize,'MarkerFaceColor','w');
p4.Color = colorWF;
p5 = plot(SNRdB,result.Nt64_N48.DIP,'--','LineWidth',linewidth);
p5.Color = colorDIP;
p6 = plot(SNRdB,result.Nt64_N48.DDP,'--','LineWidth',linewidth,'Marker','s','MarkerSize',markersize,'MarkerFaceColor','w');
p6.Color = colorDDP;

set(gca, 'FontSize',16,'FontName','Times New Roman');
h_legend = legend('Water-Filling, $N=32,L=24$', 'DIP (SGP), $N=32,L=24$', 'DDP, $N=32,L=24$', ...
                  'Water-Filling, $N=64,L=48$', 'DIP (SGP), $N=64,L=48$', 'DDP, $N=64,L=48$', ...
                  'Interpreter','latex');
legendsize = 12;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize,'FontWeight','normal','LineWidth',1,'Location','Best','NumColumns',1);
h_legend.Color = 'none';

labelsize = 19;
xlabel('Transmit SNR [dB]', 'FontSize',labelsize,'FontName','Times New Roman');
ylabel('Normalized ELMMSE [dB]', 'FontSize',labelsize,'FontName','Times New Roman');

xlim([0 50]);
ylim([-22 -8]);
xticks(0:5:50);
yticks(-22:2:-8);

grid on;
set(gca,'GridLineStyle','--','GridAlpha',0.2,'LineWidth',1,'GridLineWidth',0.5,'Layer','bottom');

set(gca,'Units','normalized');
set(gca,'Position',[0.11,0.13,0.87,0.86]);

print(gcf,'Fig13_JSAC_MATLAB.pdf','-dpdf','-vector');
hold off;


%% ========================================================================
% Local functions
% ========================================================================

function W = waterFillingPrecoder(lambda_H,Q,N,NR,sigma2_s,P)
    scale = sigma2_s*NR/N;
    targetPower = P/scale;
    muLeft = 0;
    muRight = max(1./lambda_H)+1;

    while sum(max(muRight-1./lambda_H,0)) < targetPower
        muRight = 2*muRight;
    end

    for k = 1:100
        mu = (muLeft+muRight)/2;
        powerNormalized = max(mu-1./lambda_H,0);

        if sum(powerNormalized) < targetPower
            muLeft = mu;
        else
            muRight = mu;
        end
    end

    mu = (muLeft+muRight)/2;
    powerAllocation = scale*max(mu-1./lambda_H,0);
    W = Q*diag(sqrt(powerAllocation));
end


function W = ddpPrecoderFromSVD(US,singularPower,lambda_H,Q,P,sigma2_s,NR)
    NT = length(lambda_H);
    Prev = fliplr(eye(NT));
    % Theta = 1/(sigma_s^2*N_R) * P * Sigma*Sigma^T * P
    theta = flipud(singularPower)/(sigma2_s*NR);
    active = theta > 1e-12*max(theta);

    muLeft = 0;
    muRight = 1;

    while ddpPower(muRight,theta,lambda_H,active) < P
        muRight = 2*muRight;
    end

    for k = 1:100
        mu = (muLeft+muRight)/2;

        if ddpPower(mu,theta,lambda_H,active) < P
            muLeft = mu;
        else
            muRight = mu;
        end
    end

    mu = (muLeft+muRight)/2;
    powerAllocation = zeros(NT,1);
    powerAllocation(active) = max(mu./sqrt(theta(active))-1./(lambda_H(active).*theta(active)),0);
    W = Q*diag(sqrt(powerAllocation))*Prev*US';
end


function power = ddpPower(mu,theta,lambda_H,active)
    p = zeros(length(lambda_H),1);
    p(active) = max(mu./sqrt(theta(active))-1./(lambda_H(active).*theta(active)),0);
    power = sum(p);
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


function Wproj = projectPower(W,P)
    power = norm(W,'fro')^2;
    if power <= P
        Wproj = W;
    else
        Wproj = W*sqrt(P/power);
    end
end


function J = singleLMMSE(W,S,RHinv,sigma2_s,NR)
    Delta = RHinv+1/(sigma2_s*NR)*W*S*S'*W';
    Delta = (Delta+Delta')/2;
    J = real(trace(Delta\eye(size(Delta))));
end
