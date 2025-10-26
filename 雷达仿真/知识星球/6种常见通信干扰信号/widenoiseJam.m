% produce a fragment of narrow band noise 
clear;close all;jnr = 20; % jam-noise ratio
Fs = 2000;                    % Sampling frequency
T = 1/Fs;                     % Sample time
L = 2*Fs;                     % Length of signal
t = (0:L-1)*T;                % Time vector
P=1;% power of noise
y = wgn(L,1,P); % white noise
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
WI = 50; % window of narrow band
FJ = 600;% jamming frequency
fl_kaiser = [FJ-WI/2 FJ-WI/4 FJ+WI/4 FJ+WI/2];
fl_mag = [0 1 0];
fl_dev = [0.05 0.05 0.05];
[fl_n_kaiser,fl_wn,fl_beta,fl_ftype]=kaiserord(fl_kaiser,fl_mag,fl_dev,Fs);
h= fir1(fl_n_kaiser,fl_wn,fl_ftype,kaiser(fl_n_kaiser+1,fl_beta));
Y_bp = filter(h,1,y);% the result of filter in time domain
figure;
freqz(h)
figure;
subplot(2,1,1);plot(1000*t(1:L/2),y(1:L/2));% original noise
title('White Noise');xlabel('time (milliseconds)');grid on;
subplot(2,1,2);plot(1000*t(1:L/2),Y_bp(1:L/2));% original noise
title('Narrow Band White Noise');xlabel('time (milliseconds)');grid on;

figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(1000*t(1:L/2),Y_bp(1:L/2),'Color',[0,0.4,0.8]);grid on;
title('Narrowband Noise jam')
xlabel('time (milliseconds)');


NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(Y_bp,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(f,2*10*log10(abs(Y(1:NFFT/2+1))),'Color',[0,0.4,0.8]);grid on;
title('Single-Sided Amplitude Spectrum of Narrowband Noise jam')
xlabel('Frequency (Hz)');ylabel('|Y(f)|(dB)');grid on;
% ylim([10e-4,1])


