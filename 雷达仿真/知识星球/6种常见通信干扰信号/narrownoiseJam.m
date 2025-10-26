% produce a fragment of broad band noise 
clear;close all;
Fs = 2000;                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = 2*Fs;                     % Length of signal
t = (0:L-1)*T;                % Time vector
P=1;% power of noise
y = wgn(L,1,P/600); % white noise
% 
% y = awgn(x,jnr,'measured');    % Sinusoids plus noise
figure;plot(1000*t(1:L/2),y(1:L/2));
title('White Noise')
xlabel('time (milliseconds)');grid on;

NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(y,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;plot(f,2*10*log10(abs(Y(1:NFFT/2+1)))) 
title('Single-Sided Amplitude Spectrum of White Noise')
xlabel('Frequency (Hz)');ylabel('|Y(f)|(dB)');grid on;
% ylim([10e-4,1])

%% bandpass filter
WI = 600; % window of broad band
fl_kaiser = [WI WI+50];
fl_mag = [1 0];
fl_dev = [0.05 0.01];
[fl_n_kaiser,fl_wn,fl_beta,fl_ftype]=kaiserord(fl_kaiser,fl_mag,fl_dev,Fs);
h= fir1(fl_n_kaiser,fl_wn,fl_ftype,kaiser(fl_n_kaiser+1,fl_beta));
Y_bp = filter(h,1,y);% the result of filter in time domain
figure;freqz(h);
figure;
subplot(2,1,1);plot(1000*t(1:L/2),y(1:L/2));% original noise
title('White Noise');xlabel('time (milliseconds)');grid on;
subplot(2,1,2);plot(1000*t(1:L/2),Y_bp(1:L/2));% original noise
title('Broad Band White Noise');xlabel('time (milliseconds)');grid on;

NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(Y_bp,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;plot(f,2*10*log10(abs(Y(1:NFFT/2+1)))) 
title('Single-Sided Amplitude Spectrum of Narrow band White Noise')
xlabel('Frequency (Hz)');ylabel('|Y(f)|(dB)');grid on;
% ylim([10e-4,1])



