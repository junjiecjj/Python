%Perf. of PSK/QAM/PAM over Rayleigh flat fading(complex baseband)
clearvars; clc;
%---------Input Fields------------------------
nSym=10^5;%Number of symbols to transmit
EbN0dB = -10:2:20; % Eb/N0 range in dB for simulation
MOD_TYPE='QAM'; %Set 'PSK' or 'QAM' or 'PAM' (FSK not supported)
arrayOfM=[2,4,8,16,32]; %array of M values to simulate
%arrayOfM=[4,16,64,256]; %uncomment this line if MOD_TYPE='QAM'
plotColor =['b','g','r','c','m','k']; p=1; %plot colors 
legendString = cell(1,length(arrayOfM)*2); %for legend entries
for M = arrayOfM
    %-----Initialization of various parameters----
    k=log2(M); EsN0dB = 10*log10(k)+EbN0dB; %EsN0dB calculation
    SER_sim = zeros(1,length(EbN0dB));%simulated Symbol error rates
    
    d=ceil(M.*rand(1,nSym));%uniform random symbols from 1:M
    s=modulate(MOD_TYPE,M,d);%(Refer Chapter 3)    
    for i=1:length(EsN0dB),
        h = 1/sqrt(2)*(randn(1,nSym)+1i*randn(1,nSym));%flat Rayleigh
        hs = abs(h).*s; %flat fading effect on the modulated symbols
        r = add_awgn_noise(hs,EsN0dB(i));%(Refer 4.1.2)
        
        %-----------------Receiver----------------------
        y = r./abs(h); %decision vector
        dCap  = demodulate(MOD_TYPE,M,y);%(Refer Chapter 3)
        SER_sim(i) = sum((d~=dCap))/nSym;%SER computation
    end
    SER_theory = ser_rayleigh(EbN0dB,MOD_TYPE,M); %theoretical SER
    semilogy(EbN0dB,SER_sim,[plotColor(p) '*']); hold on;
    semilogy(EbN0dB,SER_theory,plotColor(p)); 
    legendString{2*p-1}=['Sim ',num2str(M),'-',MOD_TYPE];
    legendString{2*p}=['Theory ',num2str(M),'-',MOD_TYPE]; p=p+1;
end
legend(legendString);xlabel('Eb/N0(dB)');ylabel('SER (Ps)');
title(['Probability of Symbol Error for M-',MOD_TYPE,' over Rayleigh flat fading channel']);