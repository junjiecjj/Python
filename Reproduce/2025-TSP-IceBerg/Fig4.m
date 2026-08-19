%% Fig. 4: SC versus OFDM for M=1 and M=100
clear; close all; clc;
Tsym=1; N=128; L=10; alpha=.3; span=N; K=L*N; kappa=1.32;
pulseMethod='commpy';
[p,~,~]=makeSRRCPulse(pulseMethod,alpha,L,span,Tsym,K);
Fp=fft(p)/sqrt(K); g=N*Fp.*conj(Fp); g0=g(1:N);
FN=fft(eye(N))/sqrt(N); n=(0:N-1).';
iceberg=zeros(1,K); ofdm1=zeros(1,K); ofdm100=zeros(1,K);
sc1=zeros(1,K); sc100=zeros(1,K); V=eye(N)'*FN'; Vtilde=V.*conj(V);
for kIdx=1:K
    kk=kIdx-1; gk=g0+(1-g0)*exp(-1j*2*pi*kk/L);
    fk=exp(-1j*2*pi*kk*n/K); r1=abs(gk.'*conj(fk))^2;
    q=real(N-2*(1-cos(2*pi*kk/L))*sum(g0.*(1-g0)));
    iceberg(kIdx)=r1; ofdm1(kIdx)=r1+(kappa-1)*q;
    ofdm100(kIdx)=r1+(kappa-1)*q/100;
    fcol=exp(-1j*2*pi*(0:K-1).'*kk/K)/sqrt(K); ft=fcol(1:N);
    R1=K*abs(ft'*gk)^2; R2=norm(gk)^2;
    R3=(kappa-2)*K*norm(Vtilde*(gk.*conj(ft)))^2;
    sc1(kIdx)=real(R1+R2+R3); sc100(kIdx)=real(R1+(R2+R3)/100);
end
db=@(z) 10*log10(fftshift(real(z)/max(real(z))+1e-10)); x=-K/2:K/2-1;
figure('Color','w'); hold on; grid on; box on;
plot(x,db(iceberg),'k--','LineWidth',1.5);
plot(x,db(ofdm1),'b--','LineWidth',1.5); plot(x,db(ofdm100),'b-','LineWidth',1.5);
plot(x,db(sc1),'r--','LineWidth',1.5); plot(x,db(sc100),'r-','LineWidth',1.5);
xlim([-200 200]); xlabel('Delay Index'); ylabel('Ambiguity Level (dB)');
legend('Iceberg','OFDM, M=1','OFDM, M=100','SC, M=1','SC, M=100','Location','best');
exportgraphics(gcf,'Fig4.png','Resolution',300);
