% produce a fragment of multiple sinusoid jam
clear;close all;jnr = 200; % jam-noise ratio
Q = 10; % number of multiple sinusoid
Pl = 1-0.1*rand(1,Q);                       % Jam power

Fs = 2000;                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = 2*Fs;                     % Length of signal
t = (0:L-1)*T;                % Time vector
% Sum of a 50 Hz sinusoid
fs0 = 100;deltafs = 10;
fs = fs0:deltafs:fs0+(Q-1)*deltafs; % produce same-distance fs
theta = 2*pi*rand(1,Q);% random phase
xx = cos(2*pi*fs'*t+theta'*ones(1,length(t)));% cos
x = sqrt(Pl)*xx; % multi frequency Sinusoids jam with random phase

x_mid = (max(x)+min(x))/2;
x_nor = 2*(x-x_mid)/(max(x)-min(x));% normalize the data into [-1,1]
x_dec = (x_nor-mean(x_nor))/sqrt(var(x_nor));% decentralize the data

figure;
subplot(2,1,1);
plot(Fs*t(1:5*Fs/fs(1)),x(1:5*Fs/fs(1)))
title('original data')
xlabel('time (milliseconds)');grid on;
subplot(2,1,2);
plot(Fs*t(1:5*Fs/fs(1)),x_dec(1:5*Fs/fs(1)))
title('pre_processed data')
xlabel('time (milliseconds)');grid on;

y = awgn(x,jnr,'measured');    % Sinusoids plus noise

figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(Fs*t(1:10*Fs/fs(1)),y(1:10*Fs/fs(1)),'Color',[0,0.4,0.8]);grid on;
title('multiple-tone jam')
xlabel('time (milliseconds)');
NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(y,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);

% Plot single-sided amplitude spectrum.
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
semilogy(f,2*abs(Y(1:NFFT/2+1)),'Color',[0,0.4,0.8]);grid on;
title('Single-Sided Amplitude Spectrum of multiple-tone jam');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
ylim([10e-4,1]);grid on;

