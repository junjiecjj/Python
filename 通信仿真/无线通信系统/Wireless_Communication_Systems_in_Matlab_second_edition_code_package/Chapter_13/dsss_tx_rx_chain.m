%----PRBS definition----
prbsType='MSEQUENCE'; %PRBS type
G1=[1 0 0 1 0 1];%LFSR polynomial
X1=[0 0 0 0 1];%initial seed for LFSR
G2=0;X2=0;%G2,X2 are zero for m-sequence (only one LFSR used)
%--- Input data, data rate, chip rate------
N=10e5; %number of data bits to transmit
d= rand(1,N) >=0.5; %random data
Rb=2e3; %data rate (bps) for the data d
Rc=6e3; %chip-rate(Rc >> Rb AND Rc is integral multiple of Rb)
L=8; %oversampling factor for waveform generation
SNR_dB = -4:3:20; %signal to noise ratios (dB)
BER = zeros(1,length(SNR_dB)); %place holder for BER values
for i=1:length(SNR_dB),
   [s_t,carrier_ref,prbs_ref]=dsss_transmitter(d,prbsType,G1,G2,X1,X2,Rb,Rc,L);%Tx
   %-----Compute and add AWGN noise to the transmitted signal---
   Esym=L*sum(abs(s_t).^2)/(length(s_t));%Calculate symbol energy
   N0=Esym/(10^(SNR_dB(i)/10)); %Find the noise spectral density
   n_t = sqrt(N0/2)*randn(length(s_t),1);%computed noise
   r_t = s_t + n_t; %received signal
   
   dCap = dsss_receiver(r_t,carrier_ref,prbs_ref,Rb,Rc,L);%Receiver
   BER(i) = sum(dCap~=d)/length(d); %Bit Error Rate
end
theoreticalBER = 0.5*erfc(sqrt(10.^(SNR_dB/10)));
figure; semilogy(SNR_dB, BER,'k*'); hold on;
semilogy(SNR_dB, theoreticalBER,'r');