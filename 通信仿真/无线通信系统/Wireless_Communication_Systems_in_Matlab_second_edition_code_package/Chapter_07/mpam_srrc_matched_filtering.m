%Performance simulation of MPAM with SRRC pulse shaping
clear; clc; close all;
N = 10^6; %Number of symbols to transmit
MOD_TYPE = 'QAM'; %modulation type is 'PAM'
M = 16; %modulation level for the chosen modulation MOD_TYPE
EbN0dB = -4:2:16; %SNRs for generating AWGN channel noise
%----Pulse shaping filter definitions-----
beta = 0.3;% roll-off factor for SRRC filter
Nsym = 8;%SRRC filter span in symbol durations
L = 4; %Oversampling factor (L samples per symbol period)
[p,t,filtDelay] = srrcFunction(beta,L,Nsym);%SRRC filter
%p = p/norm(p);
SER=zeros(1,length(EbN0dB)); %Symbol Error Rate place holders
snr = 10*log10(log2(M))+EbN0dB; %Converting given Eb/N0 dB to SNR

for i=1:length(snr),%simulate the system for each given SNR 
  %-------- Transmitter -------------------
  d = ceil(M.*rand(1,N));%random numbers from 1 to M for input to PAM
  u = modulate(MOD_TYPE,M,d); %modulation
  v = [u ;zeros(L-1,length(u))];%insert L-1 zero between each symbols
  v = v(:).';%Convert to a single stream, output is at sampling rate
  s = conv(v,p,'full');%Convolve modulated symbols with p[n]
  
  %-------- Channel -------------------
  r = add_awgn_noise(s,snr(i),L);%add AWGN noise for given SNR, r=s+w 
  
  %-------- Receiver -------------------
  vCap = conv(r,p,'full'); %convolve received signal with SRRC filter
  uCap = vCap(2*filtDelay+1:L:end-(2*filtDelay))/L;%downsample by L 
  %from filtdelay+1 position, matched filter output is scaled by L  
  dCap = demodulate(MOD_TYPE,M,uCap); %demodulation
  SER(i) = sum(dCap ~=d)/N; %Calculate symbol error rate
end
SER_theory = ser_awgn(EbN0dB,MOD_TYPE,M);%theoretical SER for PAM
semilogy(EbN0dB,SER,'r*'); hold on;
semilogy(EbN0dB,SER_theory,'k-')