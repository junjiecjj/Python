%% Fig. 3: SC, CDMA, OTFS, AFDM and OFDM, M=1
clear; close all; clc; rng(42,'twister');
Tsym=1; N=128; L=10; alpha=.3; span=N; K=L*N; kappa=1.32;
pulseMethod='commpy'; % switch to 'rcosdesign' for a toolbox comparison
[p,~,~]=makeSRRCPulse(pulseMethod,alpha,L,span,Tsym,K);
Fp=fft(p)/sqrt(K); g=N*Fp.*conj(Fp); g0=g(1:N);
FN=fft(eye(N))/sqrt(N); n=(0:N-1).';

Uofdm=FN'; Usc=eye(N); Ucdma=hadamard(N)/sqrt(N);
F32=fft(eye(32))/sqrt(32); Uotfs=kron(F32,eye(N/32));
c1=1/128; c2=4/(3*pi); nn=(0:N-1).';
L1=diag(exp(-1j*2*pi*c1*nn.^2)); L2=diag(exp(-1j*2*pi*c2*nn.^2));
A=L2*FN*L1; Uafdm=A';

mods={Uofdm,Usc,Ucdma,Uotfs,Uafdm}; curves=zeros(5,K); iceberg=zeros(1,K);
for uIdx=1:numel(mods)
    U=mods{uIdx}; V=U'*FN'; Vtilde=V.*conj(V);
    for kIdx=1:K
        kk=kIdx-1; gk=g0+(1-g0)*exp(-1j*2*pi*kk/L);
        fk=exp(-1j*2*pi*kk*n/K);
        r1=abs(gk.'*conj(fk))^2; iceberg(kIdx)=r1;
        if uIdx==1
            q=real(N-2*(1-cos(2*pi*kk/L))*sum(g0.*(1-g0)));
            curves(uIdx,kIdx)=r1+(kappa-1)*q;
        else
            fcol=exp(-1j*2*pi*(0:K-1).'*kk/K)/sqrt(K); ft=fcol(1:N);
            R1=K*abs(ft'*gk)^2; R2=norm(gk)^2;
            R3=(kappa-2)*K*norm(Vtilde*(gk.*conj(ft)))^2;
            curves(uIdx,kIdx)=real(R1+R2+R3);
        end
    end
end

% Numerical SC curve, Eq. (26), using MATLAB qammod.
Iter=1000; V=Usc'*FN'; sim=zeros(Iter,K); Order=16;
for kIdx=1:K
    kk=kIdx-1; gk=g0+(1-g0)*exp(-1j*2*pi*kk/L);
    fk=exp(-1j*2*pi*kk*n/K);
    for it=1:Iter
        s=qammod(randi([0 Order-1],N,1),Order,'gray','UnitAveragePower',true);
        sim(it,kIdx)=abs(sum(gk.*abs(V'*s).^2.*conj(fk)))^2;
    end
end
db=@(z) 10*log10(fftshift(real(z)/max(real(z))+1e-10)); x=-K/2:K/2-1;
figure('Color','w'); hold on; grid on; box on;
plot(x,db(iceberg),'k--','LineWidth',1.5);
styles={'b-','r-','g-','c--','m--'}; names={'OFDM','SC','CDMA','OTFS','AFDM'};
for ii=1:5, plot(x,db(curves(ii,:)),styles{ii},'LineWidth',1.5); end
plot(x,db(mean(sim,1)),'ro--','MarkerIndices',1:20:K,'MarkerFaceColor','none');
xlim([-200 200]); xlabel('Delay Index'); ylabel('Ambiguity Level (dB)');
legend([{'Iceberg'},names,{'SC simulation'}],'Location','best');
exportgraphics(gcf,'Fig3.png','Resolution',300);
