% produce a fragment of single sinusoid jam
clear;close all;
Pl = 1;
Fs = 1000;                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = 1000;                     % Length of signal
t = (0:L-1)*T;                % Time vector
% Sum of a 50 Hz sinusoid and a 120 Hz sinusoid
fs = 50;
x = sin(2*pi*fs*t); 
jnr = 200;
y = awgn(x,jnr,'measured');   % Sinusoids plus noise   
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(Fs*t(1:100),y(1:100),'Color',[0,0.4,0.8]);grid on;
title('single-tone jam')
xlabel('time (milliseconds)');
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(y,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);

% Plot single-sided amplitude spectrum.
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
semilogy(f,2*abs(Y(1:NFFT/2+1)),'Color',[0,0.4,0.8]);grid on;
title('Single-Sided Amplitude Spectrum of single-tone jam');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
ylim([10e-4,1]);grid on;
% C par
