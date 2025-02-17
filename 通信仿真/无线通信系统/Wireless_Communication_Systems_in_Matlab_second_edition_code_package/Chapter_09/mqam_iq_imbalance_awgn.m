%Eb/N0 Vs SER for M-QAM modulation with receiver impairments
clear all;clc;
%---------Input Fields------------------------
N=100000; %Number of input symbols
EbN0dB = -4:5:24; %Define EbN0dB range for simulation
M=64; %M-QAM modulation order
g=0.9; phi=8; dc_i=1.9; dc_q=1.7;%receiver impairments
%----------------------------------------------
k=log2(M); %Bits per symbol
EsN0dB = 10*log10(k)+EbN0dB; %Converting Eb/N0 to Es/N0
SER1 = zeros(length(EsN0dB),1);%Symbol Error rates (No compensation)
SER2 = SER1;%Symbol Error rates (DC compensation only)
SER3 = SER1;%Symbol Error rates (DC comp & Blind IQ compensation)
SER4 = SER1;%Symbol Error rates (DC comp & Pilot IQ compensation)

d=ceil(M.*rand(1,N));%random data symbols drawn from [1,2,..,M]
[s,ref]=mqam_modulator(M,d);%MQAM symbols & reference constellation
%See section 3.4 on chapter 'Digital Modulators and Demodulators 
%- Complex Baseband Equivalent Models' for function definition

for i=1:length(EsN0dB),
 r = add_awgn_noise(s,EsN0dB(i)); %see section 4.1 on chapter 4
 
 z=receiver_impairments(r,g,phi,dc_i,dc_q);%add impairments
 
 v=dc_compensation(z); %DC compensation
 y3=blind_iq_compensation(v); %blind IQ compensation
 [Kest,Pest]=pilot_iq_imb_est(g,phi,dc_i,dc_q);%Pilot based estimation
 y4=iqImb_compensation(v,Kest,Pest);%IQ comp. using estimated values

 %Enable this section - if you want to plot constellation diagram - 
 %figure(1);plot(real(z),imag(z),'rO'); hold on; 
 %plot(real(y4),imag(y4),'b*'); hold off;
 %title(['Eb/N0=',num2str(EbN0dB(j)),' (dB)']);pause;
 
 %-------IQ Detectors - defined in section 3.5.4 chapter 3--------
 [estTxSymbols_1,dcap_1]= iqOptDetector(z,ref);%No compensation
 [estTxSymbols_2,dcap_2]= iqOptDetector(v,ref);%DC compensation only
 [estTxSymbols_3,dcap_3]= iqOptDetector(y3,ref);%DC & blind IQ comp.
 [estTxSymbols_4,dcap_4]= iqOptDetector(y4,ref);%DC & pilot IQ comp.
 %------ Symbol Error Rate Computation-------
 SER1(i)=sum((d~=dcap_1))/N;    SER2(i)=sum((d~=dcap_2))/N;
 SER3(i)=sum((d~=dcap_3))/N;    SER4(i)=sum((d~=dcap_4))/N;
end
theoreticalSER = ser_awgn(EbN0dB,'MQAM',M); %theoretical SER
figure(2);semilogy(EbN0dB,SER1,'r*-'); hold on; %Plot results
semilogy(EbN0dB,SER2,'bO-');   semilogy(EbN0dB,SER3,'g^-');
semilogy(EbN0dB,SER4,'m*-');   semilogy(EbN0dB,theoreticalSER,'k');
legend('No compensation','DC comp only',...
 'Sim- DC & blind iq comp','Sim- DC & pilot iq comp','Theoretical');
xlabel('E_b/N_0 (dB)');ylabel('Symbol Error Rate (Ps)');
title('Probability of Symbol Error 64-QAM signals');