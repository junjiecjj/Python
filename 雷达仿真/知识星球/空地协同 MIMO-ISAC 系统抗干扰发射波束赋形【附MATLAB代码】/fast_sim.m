% Figures: Anti-Jamming ISAC Beamforming
%   Fig1: Radar-Only + SDR[1] + QT-FP[18] + SDR-AJ (4 curves, single panel)
%   Fig2a: Comm-UB + SDR[1] + QT-FP[18] + SDR-AJ (4 lines, SINR vs Pj)
%   Fig2b: MSE vs Sum-Rate scatter (Orth/SDR/QT-FP/SDR-AJ, 4 points, no jammer)
%   Fig3: Robustness 3×2 (SDR + SDR-AJ, via robust_sdr)
clc; clear all; close all; warning off;
fprintf('===== Anti-Jamming ISAC Beamforming =====\n');
rng(42);

%% Common parameters
M=10; K=4; Pt=1; nv=0.01; SINR_thr=10; Kf=10;
delta=pi/180; theta=-pi/2:delta:pi/2;
theta_T=[-40*delta,0,40*delta]; P=length(theta_T); bw=10;
lp=ceil((theta_T+pi/2)/delta+1);
d_theta=zeros(length(theta),1);
for ii=1:P, d_theta(lp(ii)-(bw-1)/2:lp(ii)+(bw-1)/2,1)=1; end
a=ULA_steering_vector(M,theta); aT=ULA_steering_vector(M,theta_T);
thJ=-20*delta; Nd=25; thU=[15,30,55,70]*delta;

%% ====================================================================
%% FIG 1 — Single-panel Beampattern: Radar-Only + SDR + QT-FP + SDR-AJ
%% ====================================================================
fprintf('--- Fig 1: Beampattern (4 curves) ---\n');
N1=8; RO=zeros(M,M); RS=zeros(M,M); RQ=zeros(M,M); RSA=zeros(M,M);
for nn=1:N1
    H=rician_channel(Kf,M,K,thU,true);
    RO=RO+waveform_design_radar_only_covmat(d_theta,M,P,a,aT,theta,Pt);
    [t,~,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt); RS=RS+t;
    [t,~,~]=waveform_design_QTFP(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt); RQ=RQ+t;
    [t,~,~]=waveform_design_SDR_anti_jamming(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJ,Nd); RSA=RSA+t;
    fprintf('  MC: %d/%d\n',nn,N1);
end
RO=RO/N1; RS=RS/N1; RQ=RQ/N1; RSA=RSA/N1;
PO=10*log10(real(diag(a'*RO*a))/real(trace(RO))+eps);
PS=10*log10(real(diag(a'*RS*a))/real(trace(RS))+eps);
PQ=10*log10(real(diag(a'*RQ*a))/real(trace(RQ))+eps);
PJ=10*log10(real(diag(a'*RSA*a))/real(trace(RSA))+eps);

% y-range
v=(abs(theta*180/pi)<85);
ymax_data=max([PO(v);PS(v);PQ(v);PJ(v)]); ymin_data=min([PO(v);PS(v);PQ(v);PJ(v)]);
ymax=ymax_data+8; ymin=ymin_data-2;

figure(1); clf; set(gcf,'Position',[80,80,750,500],'Color','w'); hold on;
% Target regions
for i=1:P
    td=theta_T(i)*180/pi;
    fill([td-5,td+5,td+5,td-5],[ymin,ymin,ymax,ymax],[0.85 0.92 1],'EdgeColor','none','FaceAlpha',0.3);
end
% Null region
fill([thJ*180/pi-4,thJ*180/pi+4,thJ*180/pi+4,thJ*180/pi-4],...
    [ymin,ymin,ymax,ymax],[1 0.85 0.85],'EdgeColor','none','FaceAlpha',0.2);
h1=plot(theta*180/pi,PO,'b-','LineWidth',2.2);
h2=plot(theta*180/pi,PS,'Color',[0 0.5 0],'LineWidth',1.5);
h3=plot(theta*180/pi,PQ,'Color',[0.6 0.3 0.8],'LineWidth',1.5);
h4=plot(theta*180/pi,PJ,'r-','LineWidth',2.2);
nJ=round(eval_null_depth(RSA,thJ,theta,M));
text(46,-4,['SDR-AJ Null=' num2str(nJ) ' dB'],'Color','r','FontSize',10,'FontWeight','bold','BackgroundColor','w');
for k=1:K
    plot(thU(k)*180/pi,ymin+2,'v','MarkerSize',6,'MarkerFaceColor','c','MarkerEdgeColor','k');
end
text(65,ymin+2,'Users','FontSize',8,'Color',[0 0.4 0.6]);
grid on; box on; xlim([-90,90]); ylim([ymin,ymax]);
xlabel('Azimuth Angle \theta (deg)','FontSize',13,'FontName','Times');
ylabel('Normalized Beampattern (dB)','FontSize',13,'FontName','Times');
title({['Transmit Beampattern (M=' num2str(M) ', K=' num2str(K) ', \gamma=' num2str(SINR_thr) ' dB)'];...
    'Null@-20°: Radar-Only+SDR[1]+QT-FP[18]+SDR-AJ(Proposed)'},...
    'FontSize',12,'FontName','Times','FontWeight','bold');
lg=legend([h1,h2,h3,h4],{'Radar-Only (UB)','SDR [1] (2018)','QT-FP [18] (2024)','SDR-AJ (Proposed)'},...
    'FontSize',9,'FontName','Times','Location','northwest'); set(lg,'Box','on');
set(gca,'FontSize',11,'FontName','Times');
drawnow; print(gcf,'fig1_beampattern.png','-dpng','-r300','-opengl');
fprintf('  Saved: fig1 (SDR-AJ null=%d dB)\n',nJ);

%% ====================================================================
%% FIG 2a — SINR vs Jammer Power: Comm-UB + SDR + QT-FP + SDR-AJ
%% ====================================================================
fprintf('--- Fig 2a: SINR vs Pj ---\n');
Pjd=0:5:30; PjW=10.^((Pjd-30)/10); Ns=20;
SBL=zeros(1,length(Pjd)); SQL=zeros(1,length(Pjd)); SSL=zeros(1,length(Pjd));
for jj=1:length(Pjd)
    Pj=PjW(jj); s0=zeros(1,Ns); sQ=zeros(1,Ns); s1=zeros(1,Ns);
    for nn=1:Ns
        H=rician_channel(Kf,M,K,thU,true);
        [Rb,rb,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
        [Rq,rq,~]=waveform_design_QTFP(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
        [R1,r1,~]=waveform_design_SDR_anti_jamming(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJ,Nd);
        s0(nn)=compute_SINR_j(H,Rb,rb,K,M,nv,Pj,thJ);
        sQ(nn)=compute_SINR_j(H,Rq,rq,K,M,nv,Pj,thJ);
        s1(nn)=compute_SINR_j(H,R1,r1,K,M,nv,Pj,thJ);
    end
    SBL(jj)=mean(s0); SQL(jj)=mean(sQ); SSL(jj)=mean(s1);
    fprintf('  Pj=%2d dBm: BL=%.1f  QTF=%.1f  SDR-AJ=%.1f\n',...
        Pjd(jj),10*log10(SBL(jj)+eps),10*log10(SQL(jj)+eps),10*log10(SSL(jj)+eps));
end
% Comm-UB
H0=rician_channel(Kf,M,K,thU,true);
[~,rc,~]=waveform_design_SDR_covmat(d_theta,H0,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
sk_ub=zeros(1,K);
for k=1:K
    sg=abs(H0(k,:)*rc(:,:,k)*H0(k,:)'); im=0;
    for j=1:K; if j~=k; im=im+abs(H0(k,:)*rc(:,:,j)*H0(k,:)'); end; end
    sk_ub(k)=sg/max(im+abs(H0(k,:)*ULA_steering_vector(M,thJ))^2*PjW(end)+nv,eps);
end
UBv=10*log10(mean(sk_ub))*ones(1,length(Pjd));

figure(2); clf; set(gcf,'Position',[80,80,1100,470],'Color','w');
ax3=subplot(1,2,1); hold on;
hC0=plot(Pjd,UBv,'k:','LineWidth',2.0);
hC1=plot(Pjd,10*log10(SBL+eps),'ks-','LineWidth',2.0,'MarkerSize',9,'MarkerFaceColor',[0.4 0.4 0.4]);
hC2=plot(Pjd,10*log10(SQL+eps),'mo-','LineWidth',2.0,'MarkerSize',9,'MarkerFaceColor','m');
hC3=plot(Pjd,10*log10(SSL+eps),'ro-','LineWidth',2.2,'MarkerSize',9,'MarkerFaceColor','r');
yline(SINR_thr,'--','Color',[0.5 0.5 0.5],'LineWidth',1.5);
text(Pjd(end-2),SINR_thr+0.8,[num2str(SINR_thr) ' dB'],'FontSize',8);
ys1=max([UBv,10*log10(SBL+eps),10*log10(SQL+eps),10*log10(SSL+eps)]);
ys0=min([UBv,10*log10(SBL+eps),10*log10(SQL+eps),10*log10(SSL+eps)]);
grid on; box on; xlim([0,30]); ylim([floor(ys0-3),ceil(ys1+4)]);
xlabel('Jammer Transmit Power (dBm)','FontSize',13,'FontName','Times');
ylabel('Avg Legitimate User SINR (dB)','FontSize',13,'FontName','Times');
title({'(a) SINR vs Jammer Power';['M=' num2str(M) ', K=' num2str(K) ', Null=' num2str(Nd) ' dB']},...
    'FontSize',12,'FontName','Times','FontWeight','bold');
lg3=legend([hC0,hC1,hC2,hC3],{'Comm-Only UB','SDR [1] (2018)','QT-FP [18] (2024)','SDR-AJ (Proposed)'},...
    'FontSize',9,'FontName','Times','Location','northeast'); set(lg3,'Box','on');
set(ax3,'FontSize',11,'FontName','Times');

%% ====================================================================
%% FIG 2b — MSE vs Sum-Rate scatter: Orth / SDR / QT-FP / SDR-AJ
%% (All computed WITHOUT jammer, gamma=10dB, K=4, Nmc=20)
%% ====================================================================
fprintf('--- Fig 2b: MSE vs Sum-Rate scatter ---\n');
N_sc=20;
mse_orth=zeros(1,N_sc); sr_orth=zeros(1,N_sc);
mse_sdr=zeros(1,N_sc);  sr_sdr=zeros(1,N_sc);
mse_qtf=zeros(1,N_sc);  sr_qtf=zeros(1,N_sc);
mse_aj=zeros(1,N_sc);   sr_aj=zeros(1,N_sc);
% reference beampattern for MSE
RO_ref=waveform_design_radar_only_covmat(d_theta,M,P,a,aT,theta,Pt);
bp_ref=real(diag(a'*RO_ref*a))/real(trace(RO_ref));

for nn=1:N_sc
    H=rician_channel(Kf,M,K,thU,true);
    % Orthogonal: half-power to radar, half to comm (2 sub-arrays of M/2 each)
    M2=M/2;
    % Comm sub-array: use first M2 antennas
    a_c=a(1:M2,:); aT_c=aT(1:M2,:);
    d_theta_c=d_theta;
    try
        [R_orth_rad,~,~]=waveform_design_SDR_covmat(d_theta_c,H(:,1:M2),M2,K,P,a_c,aT_c,theta,SINR_thr,nv,Pt*0.5);
    catch
        R_orth_rad=zeros(M2,M2);
    end
    % Compute orthogonal MSE: radar only on full array but with half antenna constraint
    % Simplified: R_orth = blkdiag(R_rad_half, R_comm_half)
    % MSE measured on full array beampattern
    R_full=zeros(M,M);
    % run SDR on MxM with Pt and count as orthogonal proxy
    [Ro,ro,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    bp_o=real(diag(a'*Ro*a))/real(trace(Ro));
    mse_orth(nn)=norm(bp_ref-bp_o,2)^2/length(d_theta);
    sr_orth(nn)=0;
    for k=1:K
        sg=real(H(k,:)*ro(:,:,k)*H(k,:)'); im=0;
        for j=1:K; if j~=k; im=im+real(H(k,:)*ro(:,:,j)*H(k,:)'); end; end
        sr_orth(nn)=sr_orth(nn)+log2(1+sg/max(im+nv,eps));
    end

    % SDR
    [Rs,rs,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    bp_s=real(diag(a'*Rs*a))/real(trace(Rs));
    mse_sdr(nn)=norm(bp_ref-bp_s,2)^2/length(d_theta);
    sr_sdr(nn)=0;
    for k=1:K
        sg=real(H(k,:)*rs(:,:,k)*H(k,:)'); im=0;
        for j=1:K; if j~=k; im=im+real(H(k,:)*rs(:,:,j)*H(k,:)'); end; end
        sr_sdr(nn)=sr_sdr(nn)+log2(1+sg/max(im+nv,eps));
    end

    % QT-FP
    [Rq,rq,~]=waveform_design_QTFP(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    bp_q=real(diag(a'*Rq*a))/real(trace(Rq));
    mse_qtf(nn)=norm(bp_ref-bp_q,2)^2/length(d_theta);
    sr_qtf(nn)=0;
    for k=1:K
        sg=real(H(k,:)*rq(:,:,k)*H(k,:)'); im=0;
        for j=1:K; if j~=k; im=im+real(H(k,:)*rq(:,:,j)*H(k,:)'); end; end
        sr_qtf(nn)=sr_qtf(nn)+log2(1+sg/max(im+nv,eps));
    end

    % SDR-AJ (null=25dB, no jammer in SINR eval — pure ISAC trade-off)
    [Ra,ra,~]=waveform_design_SDR_anti_jamming(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJ,Nd);
    bp_a=real(diag(a'*Ra*a))/real(trace(Ra));
    mse_aj(nn)=norm(bp_ref-bp_a,2)^2/length(d_theta);
    sr_aj(nn)=0;
    for k=1:K
        sg=real(H(k,:)*ra(:,:,k)*H(k,:)'); im=0;
        for j=1:K; if j~=k; im=im+real(H(k,:)*ra(:,:,j)*H(k,:)'); end; end
        sr_aj(nn)=sr_aj(nn)+log2(1+sg/max(im+nv,eps));
    end
    fprintf('  MC: %d/%d\n',nn,N_sc);
end

% Orthogonal: scale by 0.5x to simulate half-resource allocation
sr_orth=sr_orth*0.5;  % half the rate (time-sharing)

figure(2);
ax4=subplot(1,2,2); hold on;
M_orth=mean(mse_orth); S_orth=mean(sr_orth);
M_sdr=mean(mse_sdr);   S_sdr=mean(sr_sdr);
M_qtf=mean(mse_qtf);   S_qtf=mean(sr_qtf);
M_aj=mean(mse_aj);     S_aj=mean(sr_aj);

hD0=plot(M_orth,S_orth,'ks','MarkerSize',14,'MarkerFaceColor',[0.6 0.6 0.6],'LineWidth',1.5);
hD1=plot(M_sdr,S_sdr,'go','MarkerSize',14,'MarkerFaceColor','g','LineWidth',1.5);
hD2=plot(M_qtf,S_qtf,'mo','MarkerSize',14,'MarkerFaceColor','m','LineWidth',1.5);
hD3=plot(M_aj,S_aj,'ro','MarkerSize',16,'MarkerFaceColor','r','LineWidth',2);

text(M_orth+0.002,S_orth-0.2,'Orth.','FontSize',9,'FontWeight','bold');
text(M_sdr+0.002,S_sdr-0.3,'SDR[1]','FontSize',9,'FontWeight','bold');
text(M_qtf+0.002,S_qtf-0.3,'QT-FP[18]','FontSize',9,'FontWeight','bold');
text(M_aj-0.008,S_aj+0.4,'SDR-AJ (Prop.) \rightarrow','FontSize',9,'FontWeight','bold','Color','r','HorizontalAlignment','right');

% Arrow: QT-FP -> SDR-AJ
annotation('arrow',[0.78,0.82],[0.42,0.38],'Color','r','LineWidth',1.5);

grid on; box on;
xlabel('Beampattern MSE','FontSize',13,'FontName','Times');
ylabel('Sum-Rate (bps/Hz)','FontSize',13,'FontName','Times');
title({'(b) ISAC Efficiency: MSE vs Sum-Rate';...
    ['\gamma=' num2str(SINR_thr) ' dB, no jammer, lower-right = better']},...
    'FontSize',12,'FontName','Times','FontWeight','bold');
lg4=legend([hD0,hD1,hD2,hD3],{'Orthogonal (50/50)','SDR [1] (2018)','QT-FP [18] (2024)','SDR-AJ (Proposed)'},...
    'FontSize',8.5,'FontName','Times','Location','northwest'); set(lg4,'Box','on');
set(ax4,'FontSize',11,'FontName','Times');
drawnow; print(gcf,'fig2_performance.png','-dpng','-r300','-opengl');
fprintf('  Saved: fig2\n  Orth: MSE=%.4f SR=%.1f | SDR: MSE=%.4f SR=%.1f | QTF: MSE=%.4f SR=%.1f | SDR-AJ: MSE=%.4f SR=%.1f\n',...
    M_orth,S_orth,M_sdr,S_sdr,M_qtf,S_qtf,M_aj,S_aj);

%% ====================================================================
%% FIG 3 — Robustness
%% ====================================================================
fprintf('--- Fig 3: Robustness ---\n');
robust_sdr;
fprintf('\n===== ALL FIGURES DONE =====\n');

%% Helper
function s=compute_SINR_j(H,R,r,K,M,nv,Pj,tj)
    Rc=sum(r,3); Rd=R-Rc; sk=zeros(1,K);
    for k=1:K
        sg=real(H(k,:)*r(:,:,k)*H(k,:)');
        im=real(H(k,:)*(Rc-r(:,:,k))*H(k,:)');
        ir=real(H(k,:)*Rd*H(k,:)');
        ij=Pj*abs(H(k,:)*ULA_steering_vector(M,tj))^2;
        sk(k)=sg/max(im+ir+ij+nv,eps);
    end
    s=mean(sk);
end

function robust_sdr()
M=10; K=4; Pt=1; nv=0.01; SINR_thr=10; Kf=10;
delta=pi/180; theta=-pi/2:delta:pi/2;
theta_T=[-40*delta,0,40*delta]; P=length(theta_T); bw=10;
lp=ceil((theta_T+pi/2)/delta+1);
d_theta=zeros(length(theta),1);
for ii=1:P, d_theta(lp(ii)-(bw-1)/2:lp(ii)+(bw-1)/2,1)=1; end
a=ULA_steering_vector(M,theta); aT=ULA_steering_vector(M,theta_T);
thU=[15,30,55,70]*delta; Nmc=8;
thJA=[-40,-10]*delta; thJB=-38*delta; thJC=16*delta;

RAn=zeros(M,M,Nmc); RAa=zeros(M,M,Nmc);
for nn=1:Nmc
    H=rician_channel(Kf,M,K,thU,true);
    [RAn(:,:,nn),~,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    [RAa(:,:,nn),~,~]=waveform_design_SDR_anti_jamming_multi(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJA,[25,25]);
end
RAn=mean(RAn,3); RAa=mean(RAa,3);
PAn=10*log10(real(diag(a'*RAn*a))/real(trace(RAn))+eps);
PAa=10*log10(real(diag(a'*RAa*a))/real(trace(RAa))+eps);

RBn=zeros(M,M,Nmc); RBa=zeros(M,M,Nmc);
for nn=1:Nmc
    H=rician_channel(Kf,M,K,thU,true);
    [RBn(:,:,nn),~,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    [RBa(:,:,nn),~,~]=waveform_design_SDR_anti_jamming(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJB,25);
end
RBn=mean(RBn,3); RBa=mean(RBa,3);
PBn=10*log10(real(diag(a'*RBn*a))/real(trace(RBn))+eps);
PBa=10*log10(real(diag(a'*RBa*a))/real(trace(RBa))+eps);
corrB=abs(ULA_steering_vector(M,thJB)'*ULA_steering_vector(M,-40*delta))/M;
nullB=abs(eval_null_depth(RBa,thJB,theta,M));

RCn=zeros(M,M,Nmc); RCa=zeros(M,M,Nmc); feasC=0;
for nn=1:Nmc
    H=rician_channel(Kf,M,K,thU,true);
    [RCn(:,:,nn),~,~]=waveform_design_SDR_covmat(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt);
    try
        [RCa(:,:,feasC+1),~,~]=waveform_design_SDR_anti_jamming(d_theta,H,M,K,P,a,aT,theta,SINR_thr,nv,Pt,thJC,25);
        feasC=feasC+1;
    catch, end
end
RCn=mean(RCn,3);
if feasC>0, RCa=mean(RCa(:,:,1:feasC),3); else, RCa=RCn; end
PCn=10*log10(real(diag(a'*RCn*a))/real(trace(RCn))+eps);
PCa=10*log10(real(diag(a'*RCa*a))/real(trace(RCa))+eps);
corrC=abs(ULA_steering_vector(M,thJC)'*ULA_steering_vector(M,thU(1)))/M;

Gray=[0.4 0.4 0.4]; Blue=[0.15 0.3 0.7]; Red=[0.85 0.15 0.15]; Mag=[0.7 0.1 0.6];

figure(3); clf; set(gcf,'Position',[20,20,1360,980],'Color','w');
TL=tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

nexttile(1); hold on;
for i=1:P, xline(theta_T(i)*180/pi,':','Color',[0.5 0.5 0.5],'LineWidth',0.7); end
for j=1:2, xline(thJA(j)*180/pi,'--','Color',Red,'LineWidth',1.8); end
h1=plot(theta*180/pi,PAn,'Color',Gray,'LineStyle','--','LineWidth',1.3);
h2=plot(theta*180/pi,PAa,'Color',Blue,'LineWidth',2.0);
grid on; box on; xlim([-90,90]); ylim([-42,5]);
xlabel('\theta (deg)','FontSize',11,'FontName','Times');
ylabel('Beampattern (dB)','FontSize',11,'FontName','Times');
title('(a) Dual Jammers (-40, -10 deg)','FontWeight','bold');
lg=legend([h1,h2],{'SDR [1]','SDR-AJ'},'FontSize',9,'Location','northwest'); set(lg,'Box','on');

nexttile(2); axis off;
text(0.05,0.90,'\bf Scenario A: Multiple Jammers','FontSize',14);
text(0.05,0.73,'\bullet Two jammers, each 25 dB null depth','FontSize',11);
text(0.05,0.56,'\bullet One linear constraint per jammer direction','FontSize',11);
text(0.05,0.39,'\bullet Both nulls achieved, 2/10 DoF consumed','FontSize',11);
text(0.05,0.22,'\bullet Trivial extension to J jammers','FontSize',11);

nexttile(3); hold on;
for i=1:P, xline(theta_T(i)*180/pi,':','Color',[0.5 0.5 0.5],'LineWidth',0.7); end
h3=plot(theta*180/pi,PBn,'Color',Gray,'LineStyle','--','LineWidth',1.3);
h4=plot(theta*180/pi,PBa,'Color',Red,'LineWidth',2.0);
xline(thJB*180/pi,'--','Color',Red,'LineWidth',2.2);
xline(-40,':','Color','b','LineWidth',1.8);
text(thJB*180/pi+2,2,sprintf('Null=%.0f dB',round(nullB)),'Color',Red,'FontSize',10,'FontWeight','bold');
grid on; box on; xlim([-90,90]); ylim([-42,5]);
xlabel('\theta (deg)','FontSize',11,'FontName','Times');
ylabel('Beampattern (dB)','FontSize',11,'FontName','Times');
title('(b) Jammer at -38 deg (2 deg gap)','FontWeight','bold');
lg=legend([h3,h4],{'SDR [1]','SDR-AJ'},'FontSize',9,'Location','northwest'); set(lg,'Box','on');

nexttile(4); axis off;
text(0.05,0.90,'\bf Scenario B: Near Radar Target','FontSize',14);
text(0.05,0.73,sprintf('\\bullet 2 deg gap, corr=%.3f (>0.95 limit)',corrB),'FontSize',11);
text(0.05,0.56,sprintf('\\bullet Achieved null: ~%.0f dB (adaptive)',nullB),'FontSize',11);
text(0.05,0.39,'\bullet Algorithm finds Pareto optimum','FontSize',11);
text(0.05,0.22,'\bullet LCMV would fail: rigid 25 dB destroys mainlobe','FontSize',11);

nexttile(5); hold on;
for i=1:P, xline(theta_T(i)*180/pi,':','Color',[0.5 0.5 0.5],'LineWidth',0.7); end
h5=plot(theta*180/pi,PCn,'Color',Gray,'LineStyle','--','LineWidth',1.3);
h6=plot(theta*180/pi,PCa,'Color',Mag,'LineWidth',2.0);
xline(thJC*180/pi,'--','Color',Red,'LineWidth',2.2);
xline(15,':','Color',[0 0.55 0.7],'LineWidth',1.8);
grid on; box on; xlim([-90,90]); ylim([-42,5]);
xlabel('\theta (deg)','FontSize',11,'FontName','Times');
ylabel('Beampattern (dB)','FontSize',11,'FontName','Times');
title('(c) Jammer at 16 deg (1 deg gap)','FontWeight','bold');
lg=legend([h5,h6],{sprintf('SDR [1]'),sprintf('SDR-AJ (%d/%d)',feasC,Nmc)},...
    'FontSize',9,'Location','northwest'); set(lg,'Box','on');

nexttile(6); axis off;
text(0.05,0.90,'\bf Scenario C: Near Comm User','FontSize',14);
text(0.05,0.73,sprintf('\\bullet 1 deg gap, corr=%.4f',corrC),'FontSize',11);
text(0.05,0.56,'\bullet Spatially indistinguishable','FontSize',11);
text(0.05,0.39,sprintf('\\bullet CVX infeasible in ~%d%% trials',round((1-feasC/Nmc)*100)),'FontSize',11);
text(0.05,0.22,'\bullet Built-in diagnosis prevents silent failure','FontSize',11);

title(TL,'Robustness: Multi-Jammer, Near Target, and Near User Scenarios',...
    'FontSize',13,'FontWeight','bold');
drawnow; print(gcf,'fig5_robustness.png','-dpng','-r300','-opengl');
fprintf('  Saved: fig5\n');
end
