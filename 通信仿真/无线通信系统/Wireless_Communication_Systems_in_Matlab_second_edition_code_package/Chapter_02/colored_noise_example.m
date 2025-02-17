%-- Colored noise generating with Jakes PSD---
fd=10; %maximum Doppler frequency
Fs=100;%sampling frequency
N=512;%order of Jakes filter
Ts=1/Fs;%sampling interval
[h] = Jakes_filter(fd,Ts,512);%design Jakes filter

%--Exciting the filter with white noise---
x=randn(1,10000); %Generate white noise of 10,000 samples
y = conv(x,h,'valid');%convolve with Jakes filter

%--Plot time domain and freq domain view of colored noise y
subplot(1,2,1); plot(log10(abs(y))); axis tight; %Noise envelope
xlabel('n'); ylabel('y[n]');title('Noise samples')

[fftVals,freqVals]=freqDomainView(y,1/Ts,'double');
S_yy = (Ts/length(y))*abs(fftVals).^2;
subplot(1,2,2); plot(freqVals,S_yy);%PSD of generated noise
title('Noise spectrum');xlabel('f');ylabel('|S_{yy}(f)|^2');