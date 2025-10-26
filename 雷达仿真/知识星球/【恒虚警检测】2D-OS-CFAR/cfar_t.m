%Detection threshold determination
%%
%Calculates the detection threshold numerically, according to
%desired probability of false alarm with available SNR
%%
%v0: 19.12.2022
%Turker

clear all
close all
clc

%% Inputs
numsamp=1e6; %number of test samples
SNR_dB=10; %snr of signal in dB
numbins=1000; %number of bins for pdf calculation
Pfa=0.01 %desired false alarm rate

%% Signal Generation
SNR_pow=10.^(0.1*SNR_dB); %snr for power

s=ones(numsamp,1).*exp(j*2*pi*rand(numsamp,1)) * sqrt(SNR_pow); %signal without noise
n=(randn(numsamp,1)+1j*randn(numsamp,1))/sqrt(2); %noise

n_pow=sum(sum(abs(n).^2)); %power of noise
s_pow=sum(sum(abs(s).^2)); %power of signal
SNR_dB_Check=10*log10(s_pow/n_pow); %SNR check, should be equal to SNR_dB

splusn=s+n; %noisy signal

%% Probability Calculation
[counts0,centers0]=hist(abs(n),numbins); %histogram
[counts1,centers1]=hist(abs(splusn),numbins);

pdf0 = counts0 / sum(counts0); %probability density function
pdf1 = counts1 / sum(counts1);

cdf0=cumsum(pdf0); %cumulative distribution function
cdf1=cumsum(pdf1);

%% Threshold Calculation
for n=1:length(centers0)

  if (1-cdf0(n))<=Pfa
    detectthresh=centers0(n); %threshold for desired false alarm rate
    break
  end

end

for nn=1:length(centers1)

  if (centers1(nn))>=detectthresh
    Pd=1-cdf1(nn) %detection rate
    break
  end

end

%% Plots
figure;
subplot(2,1,1)
plot(centers0,pdf0,'linewidth',2)
hold on
plot(centers1,pdf1,'linewidth',2)
plot([detectthresh,detectthresh],[0,max([max(pdf0),max(pdf1)])],'linewidth',2)
grid on
legend('Noise','Signal+Noise')
ylabel('Probability Density')
xlabel('Value')
title({[strcat('SNR=',num2str(SNR_dB_Check),'dB')],[strcat('Pfa=',num2str(1-cdf0(n))),strcat(', Pd=',num2str(Pd))]})

subplot(2,1,2)
plot(centers0,cdf0,'linewidth',2)
hold on
plot(centers1,cdf1,'linewidth',2)
grid on
legend('Noise','Signal+Noise','location','southeast')
ylabel('Cumulative Distribution')
xlabel('Value')
ylim([0 1])
