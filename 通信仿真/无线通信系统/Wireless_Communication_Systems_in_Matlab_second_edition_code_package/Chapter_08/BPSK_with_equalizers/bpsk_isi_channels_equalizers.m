% Demonstration of Eb/N0 Vs SER for baseband BPSK modulation scheme
% over different ISI channels with MMSE and ZF equalizers
clear all; clc;
%---------Input Fields------------------------
N=1e5;%Number of bits to transmit
EbN0dB = 0:2:30; % Eb/N0 range in dB for simulation
M=2; %2-PSK
h_c=[0.04 -0.05 0.07 -0.21 -0.5 0.72 0.36 0.21 0.03 0.07];%Channel A
h_c=[0.407 0.815 0.407]; %uncomment this for Channel B
h_c=[0.227 0.460 0.688 0.460 0.227]; %uncomment this for Channel C
nTaps = 31; %Desired number of taps for equalizer filter

SER_zf = zeros(length(EbN0dB),1); SER_mmse = zeros(length(EbN0dB),1);

%-----------------Transmitter---------------------
d= randi([0,1],1,N); %Uniformly distributed random source symbols
ref=cos(((M:-1:1)-1)/M*2*pi); %BPSK ideal constellation
s = ref(d+1); %BPSK Mapping
x = conv(s,h_c); %apply channel effect on transmitted symbols

for i=1:length(EbN0dB),
  %----------------Channel---------------------
  r=add_awgn_noise(x,EbN0dB(i));%add AWGN noise r = x+n

  %---------------Receiver--------------------
  %DELAY OPTIMIZED MMSE equalizer
  [h_mmse,MSE,optDelay]=mmse_equalizer(h_c,EbN0dB(i),nTaps);%MMSE eq.
  y_mmse=conv(h_mmse,r);%filter the received signal through MMSE eq.
  y_mmse=y_mmse(optDelay+1:optDelay+N);%samples from optDelay position
  
  %DELAY OPTIMIZED ZF equalizer
  [h_zf,error,optDelay]=zf_equalizer(h_c,nTaps);%design ZF equalizer
  y_zf=conv(h_zf,r);%filter the received signal through ZF equalizer
  y_zf = y_zf(optDelay+1:optDelay+N);%samples from optDelay position

  %Optimum Detection in the receiver - Euclidean distance Method
  %[estimatedTxSymbols,dcap]= iqOptDetector(y_zf,ref);%See chapter 3
  %Or a Simple threshold at zero will serve the purpose    
  dcap_mmse=(y_mmse>=0);%1 is added since message symbols are m=1,2
  dcap_zf=(y_zf>=0);%1 is added since message symbols are m=1,2

  SER_mmse(i)=sum((d~=dcap_mmse))/N;%SER when filtered thro MMSE eq.
  SER_zf(i)=sum((d~=dcap_zf))/N;%SER when filtered thro ZF equalizer
end
theoreticalSER = 0.5*erfc(sqrt(10.^(EbN0dB/10)));%BPSK theory SER

figure;
semilogy(EbN0dB,SER_zf,'g'); hold on;
semilogy(EbN0dB,SER_mmse,'r','LineWidth',1.5);
semilogy(EbN0dB,theoreticalSER,'k'); ylim([1e-4,1]);
title('Probability of Symbol Error for BPSK signals');
xlabel('E_b/N_0 (dB)');ylabel('Probability of Symbol Error - P_s');
legend('ZF Equalizer','MMSE Equalizer','No interference');grid on;

[H_c,W]=freqz(h_c);%compute and plot channel characteristics
figure;subplot(1,2,1);stem(h_c);%time domain
subplot(1,2,2);plot(W,20*log10(abs(H_c)/max(abs(H_c))));%freq domain