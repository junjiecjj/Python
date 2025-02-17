clearvars; clc
snrdB=-10:0.5:30; %Range of SNRs to simulate
 
h= (randn(1,100) + 1i*randn(1,100) )/sqrt(2); %Rayleigh flat channel
sigma_z=1; %Noise power - assumed to be unity
 
snr = 10.^(snrdB/10); %SNRs in linear scale
P=(sigma_z^2)*snr./(mean(abs(h).^2)); %corresponding values for P
 
C_awgn=(log2(1+ mean(abs(h).^2).*P/(sigma_z^2)));%C_{awgn}
C_fading=mean((log2(1+ ((abs(h).^2).')*P/(sigma_z^2))));%C_{fading}
 
plot(snrdB,C_awgn,'b'); hold on; plot(snrdB,C_fading,'r'); grid on;
legend('AWGN channel capacity','Fading channel Ergodic capacity');
title('SISO fading channel - Ergodic capacity');
xlabel('SNR (dB)');ylabel('Capacity (bps/Hz)');