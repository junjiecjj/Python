clearvars; clc;
%--------Simulation parameters----------------
nSym=10^4; %Number of OFDM Symbols to transmit
EbN0dB = 0:2:20; % bit to noise ratio
MOD_TYPE='MPSK'; %modulation type - 'MPSK' or 'MQAM'
M=64; %choose modulation order for the chosen MOD_TYPE
N=64; %FFT size or total number of subcarriers (used + unused) 64
Ncp= 16; %number of symbols in the cyclic prefix

%--------Derived Parameters--------------------
k=log2(M); %number of bits per modulated symbol
EsN0dB=10*log10(k)+EbN0dB;%convert to symbol energy to noise ratio
errors= zeros(1,length(EsN0dB)); %to store symbol errors

for i=1:length(EsN0dB),%Monte Carlo Simulation
  for j=1:nSym
   %-----------------Transmitter--------------------
   d=ceil(M.*rand(1,N));%uniform distributed random syms from 1:M
   [X,ref]=modulation_mapper(MOD_TYPE,M,d);
   x=  ifft(X,N);% IDFT
   s = add_cyclic_prefix(x,Ncp); %add CP
   
   %-------------- Channel ----------------
   r = add_awgn_noise(s,EsN0dB(i));%add AWGN noise r = s+n
   
   %-----------------Receiver----------------------
   y = remove_cyclic_prefix(r,Ncp,N);%remove CP
   Y = fft(y,N);%DFT
   [~,dcap]= iqOptDetector(Y,ref);%demapper using IQ detector
   
   %----------------Error counter------------------
   numErrors=sum(d~=dcap);%Count number of symbol errors
   errors(i)=errors(i)+numErrors;%accumulate symbol errors
  end
end
simulatedSER = errors/(nSym*N);
theoreticalSER=ser_awgn(EbN0dB,MOD_TYPE,M);

%Plot theoretical curves and simulated BER points
plot(EbN0dB,log10(simulatedSER),'ro');hold on;
plot(EbN0dB,log10(theoreticalSER),'r-');grid on;
title(['Performance of ',num2str(M),'-', MOD_TYPE,' OFDM  over AWGN channel']);
xlabel('Eb/N0 (dB)');ylabel('Symbol Error Rate');
legend('simulated','theoretical');