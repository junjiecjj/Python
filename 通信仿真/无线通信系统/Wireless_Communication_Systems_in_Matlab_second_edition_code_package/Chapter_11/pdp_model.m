close all; clearvars; clc;
N=12; %number of desired taps in the TDL model

%------COST-207 model - continuous PDP model------------------
Ts = 0.1;%distance between each points in the continuous PDP (0.1us)
tau=0:Ts:7;%range of propagation delays as given in COST-207 model
pdp_continuous=1/(1-exp(-7))*exp(-tau);%COST-207(TU) continuous PDP
figure; subplot(1,2,1);
plot(tau,10*log10(pdp_continuous)); hold on;%continuous PDP

%-----COST-207 model - discrete (sampled) PDP for TDL implementation
%sample the PDP at regular intervals and use it for TDL filter model
tauMax=max(tau);
dTau=tauMax/(N-1);%step size for sampling the PDP (us)
tau=0:dTau:tauMax;%re-sampled delays
pdp_discrete = 1/(1-exp(-7))*exp(-tau); %sampled PDP (discrete)

%------Compute path gains for N-tap TDL filter--------------
p_n=zeros(1,N);%multipath PDP of TDL filter, initialize to all zeros
numFun=@(x) exp(-x)/(1-exp(-7));%given PDP equation for COST-207(TU)
p_n(1)=integral(numFun,0,dTau/2);%integration in 1st interval
for i=1:N-2,%integration for next N-2 intervals
    tau_i = i*dTau;
    p_n(i+1)=integral(numFun,tau_i-dTau/2,tau_i+dTau/2);
end
p_n(N)=integral(numFun,tauMax-dTau/2,tauMax);%last interval integral
a_n=sqrt(p_n); %computed path gains of TDL filter

disp('Tap delays tau[n] (microsecs):'); disp(num2str(tau));
disp('Tap weights a[n] : '); disp(num2str(a_n));
disp('Sampling interval(delta tau) microsecs');disp(num2str(dTau));
disp(['Sampling frequency required (MHz): ', num2str(1/dTau)]);
plotObj=stem(tau,10*log10(p_n),'r--');%plot discrete PDP
set(plotObj,'basevalue',-40);  %to show the stem plot bottom up
title('Multipath PDP');xlabel('Delay \tau(\mu s)');
ylabel('PDP(dB)'); legend('Ref. model','TDL model');

%--Frequency Correlation Function (FCF) - Fourier Transform of PDP-
nfft = 216;%FFT size
FCF_cont=fft(pdp_continuous,nfft)/length(pdp_continuous);%cont PDP
FCF_cont=abs(FCF_cont)/max(abs(FCF_cont));%normalize

FCF_tdl = fft(p_n,nfft)/length(p_n);%FT of discrete PDP 
FCF_tdl = abs(FCF_tdl)/max(abs(FCF_tdl)); %normalize

f=1/Ts/2*linspace(0,1,nfft/2+1);%compute frequency axis based on Ts
f2=1/dTau/2*linspace(0,1,nfft/2+1);%freq. axis based on dTau

%single-sided FCF of continous PDP
subplot(1,2,2);plot(f,FCF_cont(1:nfft/2+1),'b');hold on; 
%single-sided FCF of N-tap sampled PDP
plot(f2,FCF_tdl(1:nfft/2+1),'r--'); 
title('Frequency Correlation Function (FCF)');
xlabel('Frequency \nu (Hz)');ylabel('|FCF|');legend('Ref. model','TDL model');

%---------Response of the modeled filter---------
%It is assumed that the incoming signal's sampling rate is dTau
x=[zeros(1,2) 1  zeros(1,15)];%impulse input to TDL
y=filter(a_n,1,x);%filter the impulse through the designed filter
figure; stem(y,'r');%observe multipath effects of the given model
title(['Channel response of ', num2str(N), 'tap TDL model']);
xlabel('samples');ylabel('received signal amplitude');