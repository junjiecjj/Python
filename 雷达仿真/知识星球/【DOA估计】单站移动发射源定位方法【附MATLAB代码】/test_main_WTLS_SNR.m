% Y. Sun and Q. Wan, "Position Determination for Moving Transmitter Using
% Single Station," IEEE Access, vol. 6, no. 1, pp. 61103-61116, Oct. 2018.

% Figure 2 and 3

clear all; close all;
rng('default');

fc = 2.4e9; % Hz
d = 3e8/fc;   % m
wc = 3e8/fc;

senPos = [0;0];
r = 1000;    % sensor到目标的距离
doa = 112*pi/180;    % rad，sensor与目标的夹角
u = r*[cos(doa);sin(doa)]+senPos;


speed = 15; % m/s
mov_dir = 73*pi/180;
v = speed*[cos(mov_dir);sin(mov_dir)]; % m/s

K = 32;     % 累积帧数
N = 64;     % 每帧快拍数
M = 16;     % 阵元数，ULA

ts = 0:1:K-1;     % 观测时间序列
xd = (0:M-1)'*wc/2;
a = exp(1i*2*pi*rand(K,1));

mon = 1000;
SNR_dB = -21:3:21;  % dB

fs = 10e3;
T = N/fs;

% 真实的位置、角度、多普勒
posk = u + v.*ts;
doak = atan2(posk(2,:)-senPos(2),posk(1,:)-senPos(1))';
fdk = -(v'*(posk-senPos)./(sqrt(sum((posk-senPos).^2,1)))/wc)';

rng(13);
% received signal without noise
sig = zeros(N,K);
for k = 1:K
    sig(:,k) = (randn(N,1)+1i*randn(N,1))/sqrt(2);
end

rng('default');
avgD = 0; avgF = 0;
for m = 1:mon
    avgD = avgD + randn(K,1);
    avgF = avgF + randn(K,1);
end
avgD = avgD/mon;
avgF = avgF/mon;

Q1 = eye(K);
for s = 1:length(SNR_dB)
    s
    snr = 10^(SNR_dB(s)/10);

    for k = 1:K
        [ CRB ] = CRLB_AOA_Doppler( sig(:,k), doak(k), fdk(k), SNR_dB(s), a(k), fc, fs, xd );
        doaC(k) = CRB(1,1);
        freC(k) = CRB(2,2);
    end
    Qa = diag(doaC);
    Qf = diag(freC);
    Q = blkdiag(Qa,Qf);

    % CRLB
    r_k = sqrt(sum((posk-senPos).^2,1));
    for k = 1:K
        U(k,1) = -(posk(2,k)-senPos(2))/r_k(k)^2;
        U(k,2) = (posk(1,k)-senPos(1))/r_k(k)^2;
        U(k,3:4) = ts(k)*U(k,1:2);
        
        u_k = [sin(doak(k));-cos(doak(k))];
        U(k+K,:) = -1/wc*( [0,0,cos(doak(k)),sin(doak(k))] - u_k'*v*U(k,:) );
    end
    CRB = inv(U'/Q*U);
    
    % theoretical covariance
    ukT = [sin(doak),-cos(doak)];
    pk_barT = [cos(doak),sin(doak)];
    G1 = [ukT,ukT.*ts'];
    G2 = [zeros(K,2),-pk_barT];
    G = [G1;G2];
    psi = [u;v];
    for k = 1:K
        b(k) = pk_barT(k,:)*(senPos-psi(1:2)-ts(k)*psi(3:4));
    end
    B1 = diag(b);
    B2 = -diag(ukT*psi(3:4));
    B3 = wc*eye(K);
    B = [B1,zeros(K);B2,B3];
    W = eye(2*K)/(B*Q*B');
    Cov = inv(G'*W*G);
    
    % theoretical bias
    P1 = [pk_barT,ts'.*pk_barT];
    P2 = [zeros(k,2),ukT];
    H2 = (G'*W*G)\G'*W;
    H3 = P1*H2*B;
    H4 = P2*H2*B;
    q1 = diag(H3(:,1:K)*Qa);
    q2 = diag(H4(:,1:K)*Qa);
    Bias = H2*[q1;q2];
    bias_Tp(s) = sqrt(sum(Bias(1:2).^2));
    bias_Tv(s) = sqrt(sum(Bias(3:4).^2));
    
    
    uuu1 = 0; vvv1 = 0; uuu2 = 0; vvv2 = 0; uuu3 = 0; vvv3 = 0; uuu4 = 0; vvv4 = 0;
    rng('default');
    for m = 1:mon
        % 测量
        doam = doak + sqrtm(Qa)*(randn(K,1)-avgD);
        fdm = fdk + sqrtm(Qf)*(randn(K,1)-avgF);
        
        % WTLS
        [ pos, vel ] = ISA_2Sloc( senPos, fc, ts, doam, fdm, Q );
        errp1(m) = norm(u-pos,'fro')^2;
        uuu1 = uuu1 + pos;
        errv1(m) = norm(v-vel,'fro')^2;
        vvv1 = vvv1 + vel;
        
        % bias reduction
        [ pos2, vel2 ] = ISA_2SBRloc( senPos, fc, ts, doam, fdm, Q );
        errp2(m) = norm(u-pos2,'fro')^2;
        uuu2 = uuu2 + pos2;
        errv2(m) = norm(v-vel2,'fro')^2;
        vvv2 = vvv2 + vel2;
        
        % MLE
        psi = [ pos2; vel2 ];
        [ pos3, vel3 ] = ISA_MLE( senPos, fc, ts, doam, fdm, Q, psi );
        errp3(m) = norm(u-pos3,'fro')^2;
        uuu3 = uuu3 + pos3;
        errv3(m) = norm(v-vel3,'fro')^2;
        vvv3 = vvv3 + vel3;
    end
    crp1(s) = CRB(1,1)+CRB(2,2);
    crv1(s) = CRB(3,3)+CRB(4,4);
    
    msep1(s) = mean(errp1);
    biasp1(s) = norm(uuu1/mon-u,2);
    msev1(s) = mean(errv1);
    biasv1(s) = norm(vvv1/mon-v,2);
    
    msep2(s) = mean(errp2);
    biasp2(s) = norm(uuu2/mon-u,2);
    msev2(s) = mean(errv2);
    biasv2(s) = norm(vvv2/mon-v,2);
    
    msep3(s) = mean(errp3);
    biasp3(s) = norm(uuu3/mon-u,2);
    msev3(s) = mean(errv3);
    biasv3(s) = norm(vvv3/mon-v,2);
end

figure;
semilogy(SNR_dB,sqrt(msep1),'o','MarkerSize',8,'LineWidth',1.5,'DisplayName','WTLS');hold on;
semilogy(SNR_dB,sqrt(msep2),'*','MarkerSize',8,'LineWidth',1.5,'DisplayName','BR');
semilogy(SNR_dB,sqrt(msep3),'d','MarkerSize',8,'LineWidth',1.5,'DisplayName','MLE');
semilogy(SNR_dB,sqrt(crp1),'--','LineWidth',1.5,'DisplayName','CRLB');
leg1=legend('show','Location','Northeast');grid on;
set(leg1,'FontSize',11);
xlabel('SNR(dB)');ylabel('RMSE(position)');
xlim([min(SNR_dB),max(SNR_dB)]);
xticks([min(SNR_dB):6:max(SNR_dB)]);
ylim([2 1e4]);

figure;
semilogy(SNR_dB,biasp1,'o','MarkerSize',8,'LineWidth',1.5,'DisplayName','WTLS');hold on;
semilogy(SNR_dB,biasp2,'*','MarkerSize',8,'LineWidth',1.5,'DisplayName','BR');
semilogy(SNR_dB,biasp3,'d','MarkerSize',8,'LineWidth',1.5,'DisplayName','MLE');
semilogy(SNR_dB,bias_Tp,'--','MarkerSize',8,'LineWidth',1.5,'DisplayName','Thy');
leg1=legend('show','Location','Northeast');grid on;
set(leg1,'FontSize',11);
xlabel('SNR(dB)');ylabel('Bias(position)');
xlim([min(SNR_dB),max(SNR_dB)]);
xticks([min(SNR_dB):6:max(SNR_dB)]);
ylim([1e-2 1e6]);

figure;
semilogy(SNR_dB,sqrt(msev1),'o','MarkerSize',8,'LineWidth',1.5,'DisplayName','WTLS');hold on;
semilogy(SNR_dB,sqrt(msev2),'*','MarkerSize',8,'LineWidth',1.5,'DisplayName','BR');
semilogy(SNR_dB,sqrt(msev3),'d','MarkerSize',8,'LineWidth',1.5,'DisplayName','MLE');
semilogy(SNR_dB,sqrt(crv1),'--','LineWidth',1.5,'DisplayName','CRLB');
leg2=legend('show','Location','Northeast');grid on;
set(leg2,'FontSize',11);
xlabel('SNR(dB)');ylabel('RMSE(velocity)');
xlim([min(SNR_dB),max(SNR_dB)]);
xticks([min(SNR_dB):6:max(SNR_dB)]);
ylim([0.03 1e2]);

figure;
semilogy(SNR_dB,biasv1,'o','MarkerSize',8,'LineWidth',1.5,'DisplayName','WTLS');hold on;
semilogy(SNR_dB,biasv2,'*','MarkerSize',8,'LineWidth',1.5,'DisplayName','BR');
semilogy(SNR_dB,biasv3,'d','MarkerSize',8,'LineWidth',1.5,'DisplayName','MLE');
semilogy(SNR_dB,bias_Tv,'--','MarkerSize',8,'LineWidth',1.5,'DisplayName','Thy');
leg2=legend('show','Location','Northeast');grid on;
set(leg2,'FontSize',11);
xlabel('SNR(dB)');ylabel('Bias(velocity)');
xlim([min(SNR_dB),max(SNR_dB)]);
xticks([min(SNR_dB):6:max(SNR_dB)]);
ylim([1e-4 1e4]);