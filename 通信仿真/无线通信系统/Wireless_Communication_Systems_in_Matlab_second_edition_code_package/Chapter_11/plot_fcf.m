Ps = [0 -2.9 -5.8 -8.7 -11.6];%list of received powers in dBm
TAUs = [0 50e-9 100e-9 150e-9 200e-9]; %list of propagation delays

p_i = 10.^(Ps/10); %power levels in linear scale

%Plot Power Delay Profile (PDP)
subplot(1,2,1);plotObj=stem(TAUs,Ps);
set(plotObj,'basevalue',-14);  %to show the stem plot bottom up
title('Power Delay Profile');
xlabel('Relative delays \tau (s)'); ylabel('p(\tau)');

%Frequency Correlation Function (FCF)
nfft = 216;%FFT size
FCF = fft(p_i,nfft)/length(p_i);%Take FFT of PDP 
FCF = abs(FCF)/max(abs(FCF));%normalize
dTau=50e-9;%spacing of taus in the PDP
f2 = 1/dTau/2*linspace(0,1,nfft/2+1);%x-axis frequency bins
subplot(1,2,2);plot(f2,FCF(1:nfft/2+1),'r');%FCF plot
title('Frequency Correlation Function'); 
xlabel('Frequency difference \Delta f (Hz)');
ylabel('Correlation \rho(\Delta f)');