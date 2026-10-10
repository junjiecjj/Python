%% Fig. 9 of Du et al., IEEE TSP 2024: 64-QAM PCS S&C tradeoff
% Requires Optimization Toolbox (quadprog and fmincon).
% The paper does not specify the SO-CFAR reference/guard window or the exact
% sensing echo normalization. Two published endpoint Pd values are used to
% calibrate the effective echo amplitudes. Thus the Pd curves are an anchored
% reconstruction, while the PCS probabilities and AIR are computed anew.
clear; clc; close all;
rng(42);
L = 64;
NMC = 5000;
sigma2List = [0.005, 0.01];
numC0 = 13;
Pfa = 1e-4;
targetCell = 8;                         % zero-based range cell in the paper
guardCount = 2;                        % supplementary: not specified
trainingCount = 8;                     % supplementary: not specified
referencePdPSK = 0.938;                % approximately read from Fig. 9
referencePdQAM = 0.753;                % approximately read from Fig. 9
seed = 42;

levels = -7:2:7;
[inPhase, quadrature] = meshgrid(levels,levels);
xQAM = (inPhase(:)+1i*quadrature(:))/sqrt(42);
xPSK = exp(1i*2*pi*(0:63).'/64);
pUniform = ones(64,1)/64;
energy = abs(xQAM).^2;
[ringEnergy,~,ringIndex] = unique(round(energy,12));
ringCount = accumarray(ringIndex,1);
W = numel(ringEnergy);
assert(W == 9);
ringFourth = ringEnergy.^2;
energyBelow = max(ringEnergy(ringEnergy<1));
energyAbove = min(ringEnergy(ringEnergy>1));
cMin = (energyAbove-1)/(energyAbove-energyBelow)*energyBelow^2 ...
     + (1-energyBelow)/(energyAbove-energyBelow)*energyAbove^2;
cUniform = mean(energy.^2);
c0List = linspace(1.06,cUniform,numC0); % Fig. 9 shows a subset of [cMin,cUniform]
assert(abs(cMin-1.0363)<1e-3 && abs(cUniform-1.3805)<1e-3);
fprintf('c0 range: %.6f to %.6f; fourth moment of uniform QAM: %.6f\n', ...
    cMin,c0List(end),cUniform);

optionsQP = optimoptions('quadprog','Display','off', ...
    'ConstraintTolerance',1e-10,'OptimalityTolerance',1e-10);
optionsNLP = optimoptions('fmincon','Display','off','Algorithm','sqp', ...
    'ConstraintTolerance',1e-9,'OptimalityTolerance',1e-8, ...
    'StepTolerance',1e-10,'MaxIterations',250, ...
    'MaxFunctionEvaluations',3500);
Aeq = [ones(1,W);ringEnergy.';ringFourth.'];
lowerBound = zeros(W,1);
heuristicMass = zeros(W,numC0);
for j = 1:numC0
    beq = [1;1;c0List(j)];
    % P3 has infinitely many solutions for 64-QAM. Among feasible solutions,
    % use minimum squared distance from the uniform per-point distribution.
    heuristicMass(:,j) = quadprog(2*diag(1./ringCount),zeros(W,1), [],[],Aeq,beq,lowerBound,[],[],optionsQP);
    assert(~isempty(heuristicMass(:,j)));
    assert(norm(Aeq*heuristicMass(:,j)-beq,inf)<1e-6);
end

% A direct constrained maximization of P1 obtains the same target optimum
% as Algorithm 1's modified BA iteration, with stable 9-ring variables.
optimalMass = zeros(W,numC0,numel(sigma2List));
AIRopt = zeros(numel(sigma2List),numC0);
AIRheu = zeros(numel(sigma2List),numC0);
AIRpsk = zeros(numel(sigma2List),1);
AIRqam = zeros(numel(sigma2List),1);
for n = 1:numel(sigma2List)
    sigma2 = sigma2List(n);
    quadratureQAM = makeQuadrature(xQAM,sigma2,8);
    quadraturePSK = makeQuadrature(xPSK,sigma2,8);
    AIRpsk(n) = discreteAIR(pUniform,quadraturePSK);
    AIRqam(n) = discreteAIR(pUniform,quadratureQAM);
    for j = 1:numC0
        beq = [1;1;c0List(j)];
        pHeu = heuristicMass(ringIndex,j)./ringCount(ringIndex);
        AIRheu(n,j) = discreteAIR(pHeu,quadratureQAM);
        if j == numC0
            wOpt = ringCount/64;
        else
            loss = @(w) -discreteAIR(w(ringIndex)./ringCount(ringIndex), quadratureQAM);
            [wOpt,~,exitFlag] = fmincon(loss,heuristicMass(:,j), [],[],Aeq,beq,lowerBound,[],[],optionsNLP);
            if exitFlag <= 0 || norm(Aeq*wOpt-beq,inf)>1e-5
                warning('Optimization status at sigma2=%g, c0=%g: %d', ...
                    sigma2,c0List(j),exitFlag);
            end
        end
        optimalMass(:,j,n) = wOpt;
        AIRopt(n,j) = discreteAIR(wOpt(ringIndex)./ringCount(ringIndex), ...
                                 quadratureQAM);
        fprintf('sigma2=%g, c0=%.5f: AIR optimal=%.5f, heuristic=%.5f\n', ...
            sigma2,c0List(j),AIRopt(n,j),AIRheu(n,j));
    end
end

% Calibrate the two effective amplitudes from Fig. 9's PSK/QAM endpoints.
% This is essential because the paper specifies the 10-dB nominal SI ratio
% but not its scaling relative to the SO-CFAR samples after processing.
alphaCFAR = solveSOcfarScale(trainingCount,Pfa);
siAmplitude = 0.4;
for k = 1:8
    targetAmplitude = findAmplitude(referencePdPSK,xPSK,pUniform, ...
        siAmplitude,0,alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed,3);
    siAmplitude = findAmplitude(referencePdQAM,xQAM,pUniform, ...
        targetAmplitude,1,alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed,3);
end
fprintf('Calibrated effective amplitudes: target=%.6f, SI=%.6f\n', ...
    targetAmplitude,siAmplitude);
fprintf('Echo normalization is inferred from two published endpoint Pd values.\n');
PdPSK = estimatePd(xPSK,pUniform,targetAmplitude,siAmplitude, ...
    alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed);
PdQAM = estimatePd(xQAM,pUniform,targetAmplitude,siAmplitude, ...
    alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed);
PdHeu = zeros(1,numC0);
PdOpt = zeros(numel(sigma2List),numC0);
for j = 1:numC0
    pHeu = heuristicMass(ringIndex,j)./ringCount(ringIndex);
    PdHeu(j) = estimatePd(xQAM,pHeu,targetAmplitude,siAmplitude, ...
    alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed);
    for n = 1:numel(sigma2List)
        wOpt = optimalMass(:,j,n);
        pOpt = wOpt(ringIndex)./ringCount(ringIndex);
        PdOpt(n,j) = estimatePd(xQAM,pOpt,targetAmplitude,siAmplitude, ...
            alphaCFAR,L,NMC,targetCell,guardCount,trainingCount,seed);
    end
end
fprintf('Anchor check: PSK Pd=%.4f (reference %.4f), QAM Pd=%.4f (reference %.4f)\n', ...
    PdPSK,referencePdPSK,PdQAM,referencePdQAM);

width = 8;
height = 4;
fontsize = 14;
linewidth = 2;
markersize = 10;
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultTextFontName','Times New Roman');
set(groot,'defaultLegendFontName','Times New Roman');
figure(9);
set(gcf,'Units','inches','Color','white','Renderer','painters', ...
    'PaperUnits','inches','PaperPosition',[0,0,width,height], ...
    'PaperSize',[width,height]);
hold on;
hTime = plot([AIRpsk(1),AIRqam(1)],[PdPSK,PdQAM],'--', 'Color',[.62 .62 .62],'LineWidth',1.5);
plot([AIRpsk(2),AIRqam(2)],[PdPSK,PdQAM],'--', ...
    'Color',[.62 .62 .62],'LineWidth',1.5,'HandleVisibility','off');
hOptimal = plot(AIRopt(1,:),PdOpt(1,:),'-o', 'Color','#F65314','LineWidth',linewidth,'MarkerSize',6, 'MarkerFaceColor','none');
hHeuristic = plot(AIRheu(1,:),PdHeu,'-o', ...
    'Color','#00A1F1','LineWidth',linewidth,'MarkerSize',6, ...
    'MarkerFaceColor','none');
plot(AIRopt(2,:),PdOpt(2,:),'-o','Color','#F65314', ...
    'LineWidth',linewidth,'MarkerSize',6,'HandleVisibility','off');
plot(AIRheu(2,:),PdHeu,'-o','Color','#00A1F1', ...
    'LineWidth',linewidth,'MarkerSize',6,'HandleVisibility','off');
hPSK = plot(AIRpsk,PdPSK*ones(size(AIRpsk)),'*', ...
    'Color','#8A2BE2','LineWidth',1.5,'MarkerSize',markersize);
hQAM = plot(AIRqam,PdQAM*ones(size(AIRqam)),'*', ...
    'Color','#A9A9A9','LineWidth',1.5,'MarkerSize',markersize);
set(gca,'FontSize',16,'FontName','Times New Roman');
h_legend = legend([hTime,hOptimal,hHeuristic,hPSK,hQAM], ...
    {'Time-Sharing','Optimal PCS','Heuristic PCS','Uniform PSK', ...
     'Uniform QAM'},'Interpreter','latex');
legendsize = 13;
set(h_legend,'FontName','Times New Roman','FontSize',legendsize, ...
    'FontWeight','normal','LineWidth',1,'Location','southwest');
labelsize = 16;
xlabel('AIR (bps/Hz)','FontSize',labelsize, ...
    'FontName','Times New Roman','Interpreter','latex');
ylabel('$P_d$','FontSize',labelsize, ...
    'FontName','Times New Roman','Interpreter','latex');
xlim([4.35,6.03]); ylim([.74,.95]);
xticks(4.4:.2:6); yticks(.74:.02:.94);
grid on;
set(gca,'GridLineStyle','--','GridAlpha',.2,'LineWidth',1, 'GridLineWidth',.5,'Layer','bottom');
set(gca,'Units','normalized','Position',[.11,.12,.87,.86]);
drawnow;
print(gcf,'Fig9.png','-dpng','-r600');
print(gcf,'Fig9.pdf','-dpdf','-vector');

function quad = makeQuadrature(x,sigma2,order)
    ii = 1:order-1;
    J = diag(sqrt(ii/2),1)+diag(sqrt(ii/2),-1);
    [V,D] = eig(J);
    [nodes,idx] = sort(diag(D));
    oneDweights = V(1,idx).^2;
    [u,v] = meshgrid(nodes,nodes);
    [wu,wv] = meshgrid(oneDweights,oneDweights);
    offsets = sqrt(sigma2)*(u(:)+1i*v(:));
    weights = wu(:).*wv(:);
    Q = numel(x);
    R = numel(offsets);
    observations = reshape(offsets+x(:).',1,[]);
    logLikelihood = -abs(x(:)-observations).^2/sigma2 ...
        -log(pi*sigma2);
    quad.logLikelihood = logLikelihood;
    quad.maximum = max(logLikelihood,[],1);
    quad.scaledLikelihood = exp(logLikelihood-quad.maximum);
    quad.sourceIndex = repelem((1:Q).',R);
    quad.sampleIndex = sub2ind([Q,Q*R],quad.sourceIndex, ...
        (1:Q*R).');
    quad.weights = repmat(weights,Q,1);
end

function AIR = discreteAIR(p,quad)
    p = p(:);
    mix = p.'*quad.scaledLikelihood;
    logMixture = quad.maximum.'+log(max(mix(:),realmin));
    diagonalLogLikelihood = quad.logLikelihood(quad.sampleIndex);
    AIR = sum(p(quad.sourceIndex).*quad.weights ...
        .*(diagonalLogLikelihood-logMixture))/log(2);
end

function scale = solveSOcfarScale(K,pfa)
    probability = @(a) 2*sum(arrayfun(@(j) ...
        (K^(K+j)*gamma(K+j))/(gamma(K)*gamma(j+1)*(a+2*K)^(K+j)), ...
        0:K-1));
    scale = fzero(@(a) probability(a)-pfa,[0,200]);
end

function amplitude = findAmplitude(reference,x,p,targetAmplitude,mode, ...
    scale,L,NMC,cellIndex,guardCount,trainingCount,seed,upper)
    lo = 0; hi = upper;
    for k = 1:18
        mid = (lo+hi)/2;
        if mode == 0
            Pd = estimatePd(x,p,mid,targetAmplitude,scale,L,NMC,cellIndex, ...
                guardCount,trainingCount,seed);
            if Pd < reference, lo=mid; else, hi=mid; end
        else
            Pd = estimatePd(x,p,targetAmplitude,mid,scale,L,NMC, ...
                cellIndex,guardCount,trainingCount,seed);
            if Pd > reference, lo=mid; else, hi=mid; end
        end
    end
    amplitude = (lo+hi)/2;
end

function Pd = estimatePd(alphabet,p,targetAmplitude,siAmplitude, ...
    scale,L,NMC,cellIndex,guardCount,trainingCount,seed)
    rng(seed);
    probabilityEdges = cumsum(p(:));
    probabilityEdges(end) = 1;
    draw = rand(NMC,L);
    index = 1+sum(draw > reshape(probabilityEdges(1:end-1),1,1,[]),3);
    X = alphabet(index);
    phase = exp(-1i*2*pi*(0:L-1)*cellIndex/L);
    noise = (randn(NMC,L)+1i*randn(NMC,L))/sqrt(2);
    Y = siAmplitude*X+targetAmplitude*X.*phase+noise;
    rangeProfile = abs(ifft(Y.*conj(X),[],2)).^2;
    left = mod(cellIndex-guardCount-(1:trainingCount),L)+1;
    right = mod(cellIndex+guardCount+(1:trainingCount),L)+1;
    threshold = scale*min(mean(rangeProfile(:,left),2), ...
                          mean(rangeProfile(:,right),2));
    Pd = mean(rangeProfile(:,cellIndex+1)>threshold);
end