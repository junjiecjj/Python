clearvars; clc;
%--------Simulation parameters----------------
nSym=10^4; %Number of OFDM Symbols to transmit
EbN0dB = 0:2:20; % bit to noise ratio
MOD_TYPE='MQAM'; %modulation type - 'MPSK' or 'MQAM'
M=4; %choose modulation order for the chosen MOD_TYPE
N=64; %FFT size or total number of subcarriers (used + unused) 64
Ncp= 16; %number of symbols in the cyclic prefix
L=10; %Number of taps for the frequency selective channel model
%--------Derived Parameters--------------------
k=log2(M); %number of bits per modulated symbol
EsN0dB = 10*log10(k*N/(N+Ncp))+EbN0dB; %account for overheads
errors= zeros(1,length(EsN0dB)); %to store symbol errors

for i=1:length(EsN0dB),%Monte Carlo Simulation
  for j=1:nSym
   %-----------------Transmitter--------------------
   d=ceil(M.*rand(1,N));%uniform distributed random syms from 1:M
   [X,ref]=modulation_mapper(MOD_TYPE,M,d);
   x=  ifft(X,N);% IDFT
   s = add_cyclic_prefix(x,Ncp); %add CP
   
   %-------------- Channel ----------------
   h =1/sqrt(2)*(randn(1,L)+1i*randn(1,L)); %CIR
   H = fft(h,N);   
   hs = conv(h,s);%filter the OFDM sym through freq. sel. channel
   r = add_awgn_noise(hs,EsN0dB(i));%add AWGN noise r = s+n
   
   %-----------------Receiver----------------------
   y = remove_cyclic_prefix(r,Ncp,N);%remove CP
   Y = fft(y,N);%DFT
   V = Y./H;%equalization
   [~,dcap]= iqOptDetector(V,ref);%demapper using IQ detector
   
   %----------------Error counter------------------
   numErrors=sum(d~=dcap);%Count number of symbol errors
   errors(i)=errors(i)+numErrors;%accumulate symbol errors
  end
end
simulatedSER = errors/(nSym*N);%symbol error from simulation
%theoretical SER for Rayleigh flat-fading
theoreticalSER=ser_rayleigh(EbN0dB,MOD_TYPE,M);
semilogy(EbN0dB,simulatedSER,'ko');hold on;
semilogy(EbN0dB,theoreticalSER,'k-');grid on;
title(['Performance of ',num2str(M),'-', MOD_TYPE,' OFDM  over Freq Selective Rayleigh channel']);
xlabel('Eb/N0 (dB)');ylabel('Symbol Error Rate');
legend('simulated','theoretical');