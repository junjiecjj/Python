% fig14_corrected.m
% Corrected solver for the covariance optimization underlying Fig. 14.
% MATLAB R2016b or later. Base MATLAB only; no CVX/toolboxes required.
%
% RUN:  fig14_corrected
%
% IMPORTANT SCOPE
% 1. Solves max sensing_objective(Phi), subject to Ic(Phi)>=R0,
%    Phi>=0, trace(Phi)<=P. SMI uses the deterministic equivalent;
%    UB uses the Jensen upper bound. Both curves plot the same SMI metric.
% 2. Sensing is in nats/CPI, communication is in bits/s/Hz.
% 3. Uses projected gradient + scalar Lagrange-dual search, NOT the
%    paper's ADMM iteration. Every accepted point has an objective-gap
%    bound and an explicit feasibility check.
% 4. The exact Fig.14 channel matrices and NS are not supplied. The default
%    preserves the PREVIOUS SCRIPT'S ASSUMED channel construction (NS=4),
%    but removes its artificial calibration to Rmax=26.5. This default is
%    a numerical example, NOT an exact reproduction of the published data.
% 5. To compare on the SAME channel as a previous run, save the following
%    variables from that run AFTER constructing Hc, then set inputMatFile:
%    save('fig14_input.mat','RT','RR','Hc','P','sigma2_s','sigma2_c','NS','ND')
%    Actual author-provided matrices can be supplied in the same format.
%
% All output files are written to cfg.outputDir. Source inputs are untouched.

clear; clc; close all;

%% Configuration
cfg.seed = 3;
cfg.inputMatFile = '';         % e.g. 'fig14_input.mat'; empty = assumed model
cfg.NT = 16; cfg.NR = 16; cfg.NC = 4; cfg.ND = 4; cfg.K = 4;
cfg.NS = 4;                   % ASSUMPTION inherited from previous script
cfg.P_dBm = 40; cfg.noise_dBm = -90;
cfg.R0Set = [];               % []: actual active range; or e.g. 16:1:26
cfg.numPoints = 11;
cfg.rangeFraction = 0.98;     % avoid the singular multiplier at exact Rmax
cfg.outputDir = 'fig14_corrected_output';
cfg.runGradientCheck = true;
cfg.mcTrials = 2000;          % optional Eq.(8) check at 3 points; 0 disables

opt.innerGapTol = 1e-6;       % absolute gap in UNNORMALIZED objective units
opt.outerGapTol = 1e-4;       % nats; includes weighted-solve + dual errors
opt.feasTol = 1e-8;
opt.maxInnerIter = 20000;
opt.maxBacktrack = 70;
opt.maxBracket = 50;
opt.maxBisection = 75;
opt.armijo = 1e-4;

%% Inputs: all powers and noise variances use watts
rng(cfg.seed,'twister');
if isempty(cfg.inputMatFile)
    [RT,RR,Hc,P,sigma2_s,sigma2_c,NS,ND] = assumedChannel(cfg);
    modelNote = ['Assumed legacy channel, NS=',num2str(NS), ...
        '; no Rmax calibration; not author Fig.14 data.'];
else
    assert(exist(cfg.inputMatFile,'file')==2,'Input MAT file not found.');
    z = load(cfg.inputMatFile);
    required = {'RT','RR','Hc','P','sigma2_s','sigma2_c','NS','ND'};
    for j=1:numel(required)
        assert(isfield(z,required{j}),['Input missing: ',required{j}]);
    end
    RT=z.RT; RR=z.RR; Hc=z.Hc; P=z.P;
    sigma2_s=z.sigma2_s; sigma2_c=z.sigma2_c; NS=z.NS; ND=z.ND;
    modelNote = ['User-supplied matrices: ',cfg.inputMatFile];
end
assert(NS>=ND && NS==round(NS) && ND>=1 && ND==round(ND), ...
    'Require integer NS >= ND >= 1.');
assert(P>0 && sigma2_s>0 && sigma2_c>0,'P and noise must be positive.');
M = prepareModel(RT,RR,Hc,P,sigma2_s,sigma2_c,NS,ND);
[Qcomm,Rmax] = communicationOptimum(M);
M.Qcomm=Qcomm; M.Rmax=Rmax;

fprintf('\n%s\n',modelNote);
fprintf('Active transmit dimension = %d; ND = %d; NS = %d\n',M.r,ND,NS);
fprintf('Communication-only maximum = %.9f bps/Hz\n',Rmax);
fprintf('Sensing: nats/CPI. No axis/channel fitting is performed.\n');
if cfg.runGradientCheck
    gradientCheck(M);
end

%% Optimize the two sensing-only endpoints to an objective-gap tolerance
Qinit=eye(M.r)/M.r;
[Qs,ds] = solveWeighted(Qinit,M,'SMI',0,opt);
[Qu,du] = solveWeighted(Qinit,M,'UB',0,opt);
fprintf('\nSensing-only SMI design: SMI=%.8f, rate=%.8f, gap<=%.3e\n', ...
    sensingValue(Qs,M,'SMI'),ds.rate,ds.gap);
fprintf('Sensing-only UB  design: SMI=%.8f, rate=%.8f, gap<=%.3e\n', ...
    sensingValue(Qu,M,'SMI'),du.rate,du.gap);
fprintf('Actual possible tradeoff width: %.8f bps/Hz\n', ...
    Rmax-min(ds.rate,du.rate));

if isempty(cfg.R0Set)
    left=min(ds.rate,du.rate);
    width=max(0,Rmax-left);
    if width<1e-7
        warning('This channel has almost no communication/sensing tradeoff.');
        R0Set=left;
    else
        R0Set=linspace(left,left+cfg.rangeFraction*width,cfg.numPoints);
    end
else
    R0Set=sort(unique(cfg.R0Set(:).'));
end
assert(~isempty(R0Set) && all(isfinite(R0Set)) && all(R0Set>=0), ...
    'R0Set must contain finite nonnegative rates.');
assert(all(R0Set<=Rmax), ...
    'An R0 exceeds actual Rmax=%.9f. Change R0Set; do not rescale Hc.',Rmax);

%% Each point solves the same constrained problem for its requested R0
n=numel(R0Set);
QS=zeros(M.r,M.r,n); QU=QS;
rateSMI=zeros(1,n); rateUB=rateSMI;
smiSMI=rateSMI; smiUB=rateSMI; ubObjective=rateSMI;
gapSMI=rateSMI; gapUB=rateSMI; muSMI=rateSMI; muUB=rateSMI;
fprintf('\n R0          Rate(SMI)   SMI(SMI)    Rate(UB)    SMI(UB)    gapS/gapU\n');
for i=1:n
    [QS(:,:,i),a] = solveConstrained(M,'SMI',R0Set(i),Qs,ds,opt);
    [QU(:,:,i),b] = solveConstrained(M,'UB', R0Set(i),Qu,du,opt);
    validateCovariance(QS(:,:,i),M,R0Set(i),opt);
    validateCovariance(QU(:,:,i),M,R0Set(i),opt);
    rateSMI(i)=a.rate; rateUB(i)=b.rate;
    smiSMI(i)=sensingValue(QS(:,:,i),M,'SMI');
    smiUB(i)=sensingValue(QU(:,:,i),M,'SMI');
    ubObjective(i)=sensingValue(QU(:,:,i),M,'UB');
    gapSMI(i)=a.gap; gapUB(i)=b.gap; muSMI(i)=a.mu; muUB(i)=b.mu;
    assert(smiSMI(i)+2*opt.outerGapTol>=smiUB(i), ...
        'SMI optimization failed the feasible UB-design comparison.');
    fprintf('%10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %.1e/%.1e\n', ...
        R0Set(i),rateSMI(i),smiSMI(i),rateUB(i),smiUB(i),gapSMI(i),gapUB(i));
end
assert(all(diff(smiSMI)<=2*opt.outerGapTol), ...
    'SMI optimum increased beyond tolerance as the constraint tightened.');
assert(all(diff(ubObjective)<=2*opt.outerGapTol), ...
    'UB optimum increased beyond tolerance as the constraint tightened.');
% The UB design is optimized for UB, not for true/DE SMI. Its evaluated SMI
% need NOT be monotone. Do not sort/rewrite it to manufacture a desired curve.

%% Recover physically valid NT-by-ND precoders and full covariances
PhiSMI=zeros(size(RT,1),size(RT,1),n); PhiUB=PhiSMI;
F_SMI=zeros(size(RT,1),ND,n); F_UB=F_SMI;
for i=1:n
    PhiSMI(:,:,i)=P*M.U*QS(:,:,i)*M.U';
    PhiUB(:,:,i)=P*M.U*QU(:,:,i)*M.U';
    F_SMI(:,:,i)=recoverPrecoder(QS(:,:,i),M,ND);
    F_UB(:,:,i)=recoverPrecoder(QU(:,:,i),M,ND);
end

%% Optional Monte Carlo verification of Eq.(8), in nats/CPI
% This checks the asymptotic metric; MC noise is not used in optimization.
MC=struct();
if cfg.mcTrials>0
    oldRng=rng; rng(cfg.seed+100,'twister');
    idx=unique(round(linspace(1,n,min(3,n))));
    samples=(randn(M.r,NS,cfg.mcTrials)+1i*randn(M.r,NS,cfg.mcTrials))/sqrt(2*NS);
    MC.indices=idx;
    fprintf('\nMonte Carlo check (mean +/- 1.96 standard errors; nats/CPI):\n');
    for j=1:numel(idx)
        i=idx(j);
        [ms,es]=monteCarloSMI(QS(:,:,i),M,samples);
        [mu,eu]=monteCarloSMI(QU(:,:,i),M,samples);
        MC.SMImean(j)=ms; MC.SMIse(j)=es;
        MC.UBmean(j)=mu; MC.UBse(j)=eu;
        fprintf('R0=%.6f  SMI-design: DE %.6f, MC %.6f +/- %.6f\n', ...
            R0Set(i),smiSMI(i),ms,1.96*es);
        fprintf('             UB-design: DE %.6f, MC %.6f +/- %.6f\n', ...
            smiUB(i),mu,1.96*eu);
    end
    rng(oldRng);
end

%% Plot ACTUAL achieved rates, not R0 or artificially fitted coordinates
figure('Color','w','Position',[120 120 800 550]);
plot(rateSMI,smiSMI,'-o','Color',[0 0.63 0.95], ...
    'LineWidth',1.7,'MarkerSize',7,'MarkerFaceColor','w'); hold on;
plot(rateUB,smiUB,'--+','Color',[0.96 0.33 0.08], ...
    'LineWidth',1.7,'MarkerSize',8);
xlabel('Communication Rate [bps/Hz]'); ylabel('Sensing MI [nats/CPI]');
legend('SMI-oriented Precoding Design','UB-SMI-oriented Precoding Design', ...
    'Location','best');
set(gca,'FontName','Times New Roman','FontSize',13,'LineWidth',1);
grid on; box on;

if exist(cfg.outputDir,'dir')~=7, mkdir(cfg.outputDir); end
T=table(R0Set(:),rateSMI(:),smiSMI(:),rateUB(:),smiUB(:), ...
    gapSMI(:),gapUB(:),muSMI(:),muUB(:),'VariableNames', ...
    {'R0','Rate_SMI','SMI_SMI_nats','Rate_UB','SMI_UB_nats', ...
     'Gap_SMI','Gap_UB','Multiplier_SMI','Multiplier_UB'});
writetable(T,fullfile(cfg.outputDir,'Fig14_data.csv'));
save(fullfile(cfg.outputDir,'Fig14_results.mat'),'cfg','opt','modelNote', ...
    'RT','RR','Hc','P','sigma2_s','sigma2_c','NS','ND','Rmax','R0Set', ...
    'QS','QU','PhiSMI','PhiUB','F_SMI','F_UB','T','MC');
set(gcf,'PaperPositionMode','auto');
print(gcf,fullfile(cfg.outputDir,'Fig14_corrected.png'),'-dpng','-r200');
print(gcf,fullfile(cfg.outputDir,'Fig14_corrected.pdf'),'-dpdf','-bestfit');
fprintf('\nCompleted. Every plotted point passed feasibility and gap checks.\n');
fprintf('Files written to: %s\n',cfg.outputDir);

%% Local functions
function [RT,RR,Hc,P,s2s,s2c,NS,ND]=assumedChannel(c)
    % These are inherited modelling assumptions, NOT published Fig.14 data.
    P=10^((c.P_dBm-30)/10); s2s=10^((c.noise_dBm-30)/10); s2c=s2s;
    NS=c.NS; ND=c.ND; K=c.K;
    dT=25+10*rand(K,1); dR=25+10*rand(K,1);
    aod=30+30*rand(K,1); aoa=30+30*rand(K,1);
    sh=5.8*randn(K,1);
    pathPower=10.^(-0.1*(61.4+20*log10(dT+dR)+sh));
    AT=exp(1i*pi*(0:c.NT-1).'*sind(aod).')/sqrt(c.NT);
    AR=exp(1i*pi*(0:c.NR-1).'*sind(aoa).')/sqrt(c.NR);
    RT=herm(AT*diag(sqrt(pathPower))*AT');
    RR=herm(AR*diag(sqrt(pathPower))*AR');
    shC=5.8*randn;
    pathC=10^(-0.1*(61.4+20*log10(20)+shC));
    RTcorr=RT/(real(trace(RT))/c.NT);
    Gc=(randn(c.NC,c.NT)+1i*randn(c.NC,c.NT))/sqrt(2*c.NT);
    Hc=sqrt(pathC)*Gc*sqrtPSD(RTcorr);
end

function M=prepareModel(RT,RR,Hc,P,s2s,s2c,NS,ND)
    assert(size(RT,1)==size(RT,2) && size(RR,1)==size(RR,2), ...
        'RT and RR must be square.');
    assert(size(Hc,2)==size(RT,1),'Hc and RT dimensions disagree.');
    assert(all(isfinite(RT(:))) && all(isfinite(RR(:))) && all(isfinite(Hc(:))), ...
        'Channel matrices contain NaN/Inf.');
    assert(norm(RT-RT','fro')<=1e-9*max(norm(RT,'fro'),realmin), ...
        'RT is not Hermitian.');
    assert(norm(RR-RR','fro')<=1e-9*max(norm(RR,'fro'),realmin), ...
        'RR is not Hermitian.');
    [U,D]=eig(herm(RT)); t=real(diag(D));
    lr=real(eig(herm(RR)))/s2s;
    assert(max(t)>0 && max(lr)>0,'Sensing covariance is zero.');
    assert(min(t)>=-1e-10*max(t) && min(lr)>=-1e-10*max(lr), ...
        'RT or RR is not positive semidefinite.');
    keep=t>1e-12*max(t); U=U(:,keep); t=t(keep);
    lr=lr(lr>1e-12*max(lr));
    r=numel(t);
    assert(r<=ND, ...
        ['rank(RT)>ND: a covariance relaxation could violate stream count. ', ...
         'This implementation requires the paper example rank(RT)<=ND.']);
    outside=norm(Hc-Hc*U*U','fro')/max(norm(Hc,'fro'),realmin);
    assert(outside<1e-6, ...
        ['Hc has energy outside range(RT). Common-subspace reduction is ', ...
         'invalid; do not silently use this reduced solver.']);
    M.r=r; M.U=U; M.P=P; M.NS=NS;
    % Fold physical scales into A,C. Optimize trace(Q)=1, Phi=P*U*Q*U'.
    M.A=diag(sqrt(P*max(lr)*t)); M.w=lr/max(lr);
    M.C=sqrt(P/(NS*s2c))*Hc*U;
end

function [v,G]=sensingValue(Q,M,kind)
    B=herm(M.A*Q*M.A);
    [V,D]=eig(B); t=max(real(diag(D)),0);
    b=M.w*t.';
    if strcmp(kind,'SMI')
        % eta=1/(1+lambda*delta). Unique root on (0,1]:
        % NS*(1-eta)=sum_i eta*b_i/(1+eta*b_i).
        % This dimensionless equation avoids an absolute tolerance on a
        % tiny physical delta. Monotone bisection is robust at high SNR.
        lo=zeros(size(M.w)); hi=ones(size(M.w));
        for j=1:60
            eta=(lo+hi)/2; eb=bsxfun(@times,eta,b);
            residual=M.NS*(1-eta)-sum(eb./(1+eb),2);
            active=residual>0; lo(active)=eta(active); hi(~active)=eta(~active);
        end
        eta=(lo+hi)/2;
        v=sum(sum(log1p(bsxfun(@times,eta,b)))) ...
            -M.NS*sum(log(eta))-M.NS*sum(1-eta);
        gamma=M.w.*eta;
    elseif strcmp(kind,'UB')
        v=sum(sum(log1p(b))); gamma=M.w;
    else
        error('Unknown sensing objective.');
    end
    if nargout>1
        d=sum(bsxfun(@rdivide,gamma,1+gamma*t.'),1).';
        G=herm(M.A*V*diag(d)*V'*M.A);
    end
end

function [v,G]=communicationValue(Q,M)
    B=herm(eye(size(M.C,1))+M.C*Q*M.C');
    [L,p]=chol(B,'lower'); assert(p==0,'Communication matrix is not positive definite.');
    v=2*sum(log(real(diag(L))))/log(2);
    if nargout>1
        Z=L\M.C; G=herm(Z'*Z/log(2));
    end
end

function [v,G,s,c]=weightedValue(Q,M,kind,mu)
    [s,Gs]=sensingValue(Q,M,kind); [c,Gc]=communicationValue(Q,M);
    % Scaling changes neither the optimizer nor the dual multiplier.
    v=(s+mu*c)/(1+mu); G=(Gs+mu*Gc)/(1+mu);
end

function [Q,info]=solveWeighted(Q,M,kind,mu,opt)
    % Maximize sensing(Q)+mu*rate(Q), Q>=0, trace(Q)=1.
    % Concavity gives a global objective-gap bound:
    % max_X f(X)-f(Q) <= lambda_max(grad f)-trace(grad f*Q).
    Q=projectDensity(Q);
    [f,G,s,c]=weightedValue(Q,M,kind,mu);
    step=1/max(1,norm(G,'fro'));
    for it=0:opt.maxInnerIter
        gap=max(0,max(real(eig(herm(G))))-inner(G,Q))*(1+mu);
        if gap<=opt.innerGapTol
            info=struct('value',s,'rate',c,'gap',gap,'iterations',it,'mu',mu);
            return;
        end
        if it==opt.maxInnerIter, break; end
        accepted=false;
        for j=1:opt.maxBacktrack
            Qnew=projectDensity(Q+step*G); D=Qnew-Q;
            [fn,Gn,sn,cn]=weightedValue(Qnew,M,kind,mu);
            rounding=20*eps*max(1,abs(f));
            if fn>=f+opt.armijo*inner(G,D)-rounding
                accepted=true; break;
            end
            step=step/2;
        end
        assert(accepted,'Line search failed; no unconverged point is accepted.');
        den=inner(D,G-Gn);
        if den>1e-20
            step=min(1e6,max(1e-12,norm(D,'fro')^2/den));
        else
            step=min(1e6,1.5*step);
        end
        Q=Qnew; f=fn; G=Gn; s=sn; c=cn;
    end
    error('Weighted solve did not converge: %s, mu=%.6g, gap=%.3e.',kind,mu,gap);
end

function [Q,out]=solveConstrained(M,kind,R0,Q0,d0,opt)
    if d0.rate>=R0
        Q=Q0; out=d0; out.mu=0; return;
    end
    if R0==M.Rmax
        % At the exact maximum, water filling is the unique usable covariance
        % on the nonzero communication subspace; all available power is used.
        Q=M.Qcomm;
        out=struct('value',sensingValue(Q,M,kind),'rate',M.Rmax, ...
            'gap',0,'mu',Inf,'iterations',0);
        return;
    end
    assert(R0<M.Rmax,'Requested rate is infeasible.');
    lo=0; hi=1; Q=Q0; Qlo=Q0; bracketed=false;
    upper=d0.value+d0.gap; upperMu=0;
    for j=1:opt.maxBracket
        [Q,d]=solveWeighted(Q,M,kind,hi,opt);
        candidateUpper=d.value+hi*(d.rate-R0)+d.gap;
        if candidateUpper<upper
            upper=candidateUpper; upperMu=hi;
        end
        if d.rate>=R0
            Qhi=Q; bracketed=true; break;
        end
        lo=hi; hi=2*hi; Qlo=Q;
    end
    assert(bracketed,'Could not bracket the rate constraint.');
    bestValue=-Inf;
    for j=0:opt.maxBisection
        % For any mu>=0, optimum constrained sensing is no greater than
        % max_Q[sensing(Q)+mu*rate(Q)]-mu*R0.
        % A convex combination of Qlo and Qhi remains PSD with trace 1.
        % Find a feasible point at the rate boundary; this avoids treating
        % excess rate from an inexact weighted solve as irreducible error.
        % This interpolation is part of the constrained optimization, not
        % interpolation of plotted data. All metrics are recomputed on Q.
        Qcandidate=rateBoundary(Qlo,Qhi,R0,M);
        value=sensingValue(Qcandidate,M,kind);
        if value>bestValue
            bestValue=value; bestQ=Qcandidate;
        end
        assert(upper-bestValue>=-1e-8*max(1,abs(bestValue)), ...
            'Dual upper bound fell below feasible objective; numerical failure.');
        bound=max(0,upper-bestValue);
        if bound<=opt.outerGapTol
            Q=bestQ;
            out=struct('value',bestValue,'rate',communicationValue(Q,M), ...
                'gap',bound,'mu',upperMu,'iterations',j);
            return;
        end
        if j==opt.maxBisection, break; end
        mu=(lo+hi)/2;
        [Q,d]=solveWeighted(Q,M,kind,mu,opt);
        candidateUpper=d.value+mu*(d.rate-R0)+d.gap;
        if candidateUpper<upper
            upper=candidateUpper; upperMu=mu;
        end
        if d.rate>=R0
            hi=mu; Qhi=Q;
        else
            lo=mu; Qlo=Q;
        end
    end
    error('Constrained solve did not converge: %s, R0=%.8f, gap=%.3e.', ...
        kind,R0,bound);
end

function Q=rateBoundary(Qlo,Qhi,R0,M)
    % The superlevel set of the concave rate is convex. Along this segment
    % its feasible portion is an interval containing the high endpoint.
    lo=0; hi=1;
    for j=1:55
        t=(lo+hi)/2; Q=(1-t)*Qlo+t*Qhi;
        if communicationValue(Q,M)>=R0, hi=t; else, lo=t; end
    end
    Q=herm((1-hi)*Qlo+hi*Qhi);
end

function Q=projectDensity(X)
    % Exact Frobenius projection onto Hermitian PSD matrices with trace 1.
    [V,D]=eig(herm(X)); d=real(diag(D));
    u=sort(d,'descend'); cs=cumsum(u);
    k=find(u-(cs-1)./(1:numel(u)).'>0,1,'last');
    tau=(cs(k)-1)/k; d=max(d-tau,0);
    Q=herm(V*diag(d)*V');
end

function [Q,Rmax]=communicationOptimum(M)
    [~,S,V]=svd(M.C,'econ'); g=real(diag(S)).^2;
    assert(max(g)>0,'Communication channel is zero.');
    % Full water filling over nonzero singular values; no arbitrary Rmax fit.
    q=zeros(size(g)); active=g>0; bottom=1./g(active);
    lo=0; hi=1+min(bottom);
    for j=1:120
        level=(lo+hi)/2;
        if sum(max(level-bottom,0))<1, lo=level; else, hi=level; end
    end
    q(active)=max((lo+hi)/2-bottom,0); q=q/sum(q);
    Q=herm(V*diag(q)*V'); Rmax=communicationValue(Q,M);
end

function validateCovariance(Q,M,R0,opt)
    assert(norm(Q-Q','fro')<opt.feasTol,'Q is not Hermitian.');
    assert(min(real(eig(Q)))>=-opt.feasTol,'Q is not PSD.');
    assert(abs(real(trace(Q))-1)<opt.feasTol,'Power constraint failed.');
    assert(communicationValue(Q,M)>=R0-opt.feasTol,'Rate constraint failed.');
end

function gradientCheck(M)
    state=rng; rng(197,'twister'); Q=eye(M.r)/M.r;
    fprintf('\nFinite-difference checks (4 Hermitian directions):\n');
    names={'SMI','UB','RATE'};
    for k=1:3
        worst=0;
        for j=1:4
            D=randn(M.r)+1i*randn(M.r); D=herm(D); D=D/norm(D,'fro');
            h=1e-5/M.r;
            if k<3
                [~,G]=sensingValue(Q,M,names{k});
                fd=(sensingValue(Q+h*D,M,names{k})-sensingValue(Q-h*D,M,names{k}))/(2*h);
            else
                [~,G]=communicationValue(Q,M);
                fd=(communicationValue(Q+h*D,M)-communicationValue(Q-h*D,M))/(2*h);
            end
            predicted=inner(G,D);
            err=abs(fd-predicted)/max([1,abs(fd),abs(predicted)]);
            worst=max(worst,err);
        end
        fprintf('  %s: maximum scaled error %.3e\n',names{k},worst);
        assert(worst<1e-4,'Gradient check failed for %s.',names{k});
    end
    rng(state);
end

function [v,se]=monteCarloSMI(Q,M,S)
    L=M.A*sqrtPSD(Q); n=size(S,3); values=zeros(n,1);
    for k=1:n
        X=L*S(:,:,k); t=max(real(eig(herm(X*X'))),0);
        values(k)=sum(sum(log1p(M.w*t.')));
    end
    v=mean(values); se=std(values)/sqrt(n);
end

function F=recoverPrecoder(Q,M,ND)
    F=zeros(size(M.U,1),ND);
    F(:,1:M.r)=sqrt(M.P)*M.U*sqrtPSD(Q);
end

function A=sqrtPSD(A)
    [V,D]=eig(herm(A)); d=max(real(diag(D)),0);
    A=herm(V*diag(sqrt(d))*V');
end

function v=inner(A,B)
    v=real(sum(conj(A(:)).*B(:)));
end

function A=herm(A)
    A=(A+A')/2;
end
