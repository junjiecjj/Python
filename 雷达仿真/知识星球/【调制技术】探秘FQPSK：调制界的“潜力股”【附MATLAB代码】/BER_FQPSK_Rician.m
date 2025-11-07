clc;
clear all;
close all;
%--------Inputs-------------------------------------------------
N=1e6; %Number of data samples to send across the Rician Channel
EbN0dB=0:11; %Eb/N0 in dB overwhich the performance has to be simulated
totPower=1; %Total power of LOS path & scattered paths
K=[1 5 20 30 60 80]; %A list of Ricial K factors to simulate
%--------------------------------------------------------------
tx_data = randi([0 15],1,N);
simBER_rician=zeros(1,length(EbN0dB));
plotStyle={'b*-','r*-','k*-','g*-','m*-','c*-'};
tx = dpskmod(tx_data,16,pi/16); % 16DPSK Modulation
EbN0=10.^(EbN0dB/10); % Eb/N0 in Linear Scale
snr = 4*EbN0;         % Signal to Noise Ratio 
for index=1:length(K)
    k = K(index);
    %Derive non-centrality parameter and sigma for the underlying
    %Gaussian RVs to generate the Rician Envelope
    s=sqrt(k/(k+1)*totPower); %Non-Centrality Parameter
    sigma=totPower/sqrt(2*(k+1));
    for i=1:length(EbN0dB)
        h=((sigma*randn(1,N)+s)+1i*(randn(1,N)*sigma+0)); %Rician Fading - single tap
        %received signal through Rician & AWGN channel 
        rx_rician=h.*tx; 
        rx = awgn(rx_rician,snr(i));
        %Coherent Receiver for Rician Channel
        rx_data = dpskdemod(rx,16,pi/16);
        simBER_rician(i)=sum(xor(tx_data,rx_data));
    end
    simBER_rician=simBER_rician/N; %Error Probability
    
    %plot Simulated BER;
    semilogy(EbN0dB,simBER_rician,plotStyle{index},'LineWidth',2);
    hold on
    legendInfo{index} = ['K = ' num2str(K(index))];
end
legend(legendInfo);
title('Eb/N0 Vs BER for FQPSK over Rician Fading Channels with AWGN noise');
xlabel('Eb/N0(dB)');
ylabel('Bit Error Rate (BER)');
grid;
set(gca,'FontSize',14);
axis([0 11 1e-2 1])
