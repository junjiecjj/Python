A = 0.5; %amplitude of the cosine wave
fc=10;%frequency of the cosine wave
phase=30; %desired phase shift of the cosine in degrees
fs=32*fc;%sampling frequency with oversampling factor 32
t=0:1/fs:2-1/fs;%2 seconds duration
phi = phase*pi/180; %convert phase shift in degrees in radians
x=A*cos(2*pi*fc*t+phi);%time domain signal with phase shift

subplot(2,2,1); plot(t,x); title('time domain');
N=256; %FFT size
X = 1/N*fftshift(fft(x,N));%N-point complex DFT

df=fs/N; %frequency resolution
sampleIndex = -N/2:N/2-1; %ordered index for FFT plot
f=sampleIndex*df; %x-axis index converted to ordered frequencies
subplot(2,2,2); stem(f,abs(X)); %magnitudes vs frequencies
title('Magnitude spectrum');xlabel('f (Hz)'); ylabel('|X(k)|');

phase=atan2(imag(X),real(X))*180/pi; %raw phase information (noisy)
%plot(f,phase); %phase vs frequencies

X2=X;%store the FFT results in another array
%detect noise (very small numbers (eps)) and ignore them
threshold = max(abs(X))/10000; %tolerance threshold
X2(abs(X)<threshold) = 0; %maskout values that are below the threshold
phase=atan2(imag(X2),real(X2))*180/pi; %phase information
subplot(2,2,3); plot(f,phase); %phase vs frequencies
title('Phase spectrum');

x_recon = N*ifft(ifftshift(X),N); %reconstructed signal
t = [0:1:length(x_recon)-1]/fs; %recompute time index 
subplot(2,2,4); plot(t,x_recon);%reconstructed signal
title('Reconstructed')