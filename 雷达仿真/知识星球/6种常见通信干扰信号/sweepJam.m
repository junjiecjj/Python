% produce a fragment of scanning frequency jam
clear;close all;jnr = 1000; % jam-noise ratio
Fs = 2000;                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = 2*Fs;                     % Length of signal 2Second
t = (0:L-1)*T;                % Time vector
om = 2*pi*10;
be = 2*pi*50;
phi = 0;
x = exp(1i*0.5*be*t.^2+1i*om*t+1i*phi);
% x = exp(1i*om*t+1i*phi);

x = real(x);
y = awgn(x,jnr,'measured');    % Sinusoids plus noise
figure;plot(Fs*t,y)
title('Signal Corrupted with Zero-Mean Random Noise')
xlabel('time (milliseconds)');grid on;

figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(1000*t(1:L/2),y(1:L/2),'Color',[0,0.4,0.8]);grid on;
title('Sweeping Jam')
xlabel('time (milliseconds)');


NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(y,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(f,2*10*log10(abs(Y(1:NFFT/2+1))),'Color',[0,0.4,0.8]);grid on;
title('Single-Sided Amplitude Spectrum of Sweeping Jam')
xlabel('Frequency (Hz)');ylabel('|Y(f)|(dB)');grid on;
xlim([0,500])

% plot(2*abs(Y(1:NFFT/2+1)))



