% produce a fragment of comb-like noise 
clear;close all;
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
WI = 50; % window of broad band
FJ_set = [100,200,300,400];
for i = 1:4
    FJ = FJ_set(i);% jamming frequency
    fcuts = [FJ-WI/3 FJ-WI/4 FJ+WI/4 FJ+WI/3];
    mags = [0 1 0];
    devs = [0.02 0.02 0.02];
    [n,wn,beta,ftype]=kaiserord(fcuts,mags,devs,Fs);
    h= fir1(n,wn,ftype,kaiser(n+1,beta));
    Y_bp_temp(i,:) = filter(h,1,y);% the result of filter in time domain
end
Y_bp = sum(Y_bp_temp);
figure;
subplot(2,1,1);plot(1000*t(1:L/2),y(1:L/2));% original noise
title('White Noise');xlabel('time (milliseconds)');grid on;
subplot(2,1,2);plot(1000*t(1:L/2),Y_bp(1:L/2));% original noise
title('Partialband Noise Jam');xlabel('time (milliseconds)');grid on;

figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(1000*t(1:L/2),Y_bp(1:L/2),'Color',[0,0.4,0.8]);grid on;
title('Partialband Noise Jam')
xlabel('time (milliseconds)');

NFFT = 2^nextpow2(L); % Next power of 2 from length of y
Y = fft(Y_bp,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure;set(gcf,'Color','w','Position',[400 300 600 300]);
plot(f,2*10*log10(abs(Y(1:NFFT/2+1))),'Color',[0,0.4,0.8]);grid on;
title('Single-Sided Amplitude Spectrum of Partialband Noise Jam')
xlabel('Frequency (Hz)');ylabel('|Y(f)|(dB)');grid on;
% ylim([10e-4,1])


