%Performance of PSK/QAM/PAM over Rician flat fading
clearvars; clc;
nSym=10^6;%Number of symbols to transmit
EbN0dB = 0:2:20; % Eb/N0 range in dB for simulation
K_dB = [3,5,10,20]; %array of K factors for Rician fading in dB
MOD_TYPE='QAM'; %Set 'PSK' or 'QAM' or 'PAM' (FSK not supported)
M=64; %M value for the modulation to simulate

plotColor =['b','g','r','c','m','k']; p=1; %plot colors
legendString = cell(1,length(K_dB)*2); %for legend entries

for j = 1:length(K_dB),
    k=log2(M); EsN0dB = 10*log10(k)+EbN0dB; %EsN0dB calculation
    SER_sim = zeros(1,length(EbN0dB));%simulated Symbol error rates
    
    d=ceil(M.*rand(1,nSym));%uniform random symbols from 1:M
    s=modulate(MOD_TYPE,M,d);%(Refer Chapter 3)
    
    K = 10.^(K_dB(j)/10); %K factor in linear scale
    g1 = sqrt(K/(2*(K+1))); g2 = sqrt(1/(2*(K+1)));
    
    for i=1:length(EsN0dB),
        h = (g2*randn(1,nSym)+g1)+1i*(g2*randn(1,nSym)+g1);
        display(mean(abs(h).^2));%avg power of the fading samples 
        hs = abs(h).*s; %Rician fading effect on modulated symbols        
        r = add_awgn_noise(hs,EsN0dB(i));%(Refer 4.1.2)
        
        %-----------------Receiver----------------------
        y = r./abs(h); %decision vector
        dCap  = demodulate(MOD_TYPE,M,y);%(Refer Chapter 3)
        SER_sim(i) = sum((d~=dCap))/nSym;%symbol error rate
    end
    SER_theory = ser_rician(EbN0dB,K_dB(j),MOD_TYPE,M);%theoretical SER
    semilogy(EbN0dB,SER_sim,[plotColor(p) '*']); hold on;
    semilogy(EbN0dB,SER_theory,plotColor(p)); 
    legendString{2*p-1}=['Sim K=',num2str(K_dB(j)), ' dB'];
    legendString{2*p}=['Theory K=',num2str(K_dB(j)), ' dB']; p=p+1;
end
legend(legendString);xlabel('Eb/N0(dB)');ylabel('SER (Ps)');
title(['Probability of Symbol Error for ',num2str(M),'-',MOD_TYPE,' over Rician flat fading channel']);