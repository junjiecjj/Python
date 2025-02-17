clearvars; clc;
fc=1800E6; %Carrier frequency (Hz)
fm=200; %Maximum Doppler shift (Hz)
trms=30e-6; %RMS Delay spread (s)
Plocal = 0.2; %local-mean power (W)

f_delta=fm/30; %step size for frequency shifts
f=(fc-fm)+f_delta:f_delta:(fc+fm)-f_delta;%normalized freq shifts
tau=0:trms/5:trms*3; %generate range for propagation delays
[TAU,F]=meshgrid(tau,f);%all possible combinations of Taus and Fs

%Example Scattering function equation
Z =Plocal./(4*pi*fm*sqrt(1-((F-fc)/fm).^2)).*1/trms.*exp(-TAU/trms);
subplot(2,1,1); mesh(TAU,(F-fc)/fm,Z);%Plot the 3D mesh plot
title('Scattering function S(f,\tau)');xlabel('Delay \tau');
ylabel('(f-fc)/fm');zlabel('Received power');

%Project the 3D plot and plot PDP & Doppler Spectrum
subplot(2,2,3); plot(tau,Z(1,:,:));
title('Power Delay Profile');
xlabel('Delay (\tau)'); ylabel('Received power');
subplot(2,2,4); plot((f-fc)/fm,Z(:,1,:));
title('Doppler Power Spectrum'); 
xlabel('Doppler shift (f-fc)/fm');ylabel('Received power');