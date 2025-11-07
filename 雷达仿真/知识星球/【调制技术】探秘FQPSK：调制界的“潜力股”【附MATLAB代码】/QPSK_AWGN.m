clc;
clear all;
close all;
%--------Inputs-------------------------------------------------
N=1e6; %Number of data samples to send across the Rician Channel
EbN0dB=-5:2:15; %Eb/N0 in dB overwhich the performance has to be simulated
%--------------------------------------------------------------
tx_data = randi([0 3],1,N);
simBER = zeros(1,length(EbN0dB)); %simulated BER
tx = pskmod(tx_data,4,pi/4); % QPSK Modulation
EbN0 = 10.^(EbN0dB/10); %Eb/N0 in Linear Scale
snr = 2*EbN0; % Signal to Noise Ratio
for i=1:length(EbN0dB)
        rx=awgn(tx,snr(i)); %received signal through AWGN channel
        %Coherent Receiver for Rician Channel
        rx_data = pskdemod(rx,4,pi/4);
        simBER(i)=sum(xor(tx_data,rx_data));
    end
    simBER=simBER/N; % Error Probability 
    theoricalBER = qfunc(sqrt(2*EbN0));
    semilogy(EbN0dB,theoricalBER,'k',EbN0dB,simBER,'r*','LineWidth',2);
   
legend('QPSK theorical','QPSK simulated');
title('Eb/N0 Vs BER for QPSK');
xlabel('Eb/N0(dB)');
ylabel('Bit Error Rate (BER)');
grid;
set(gca,'FontSize',14);
axis([-2 11 1e-6 1])