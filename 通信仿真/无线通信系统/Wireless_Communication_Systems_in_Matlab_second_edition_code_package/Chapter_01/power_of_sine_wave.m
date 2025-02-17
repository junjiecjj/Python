A=1; %Amplitude of sine wave
fc=100; %Frequency of sine wave
fs=3000; %Sampling frequency - oversampled by the rate of 30
nCyl=3; %Number of cycles of the sinewave
 
t=0:1/fs:nCyl/fc-1/fs; %Time base
x=-A*sin(2*pi*fc*t); %Sinusoidal function
subplot(1,2,1); plot(t,x);title('Sinusoid of frequency f_c=100 Hz');
xlabel('Time(s)'); ylabel('Amplitude');

L=length(x);
P=(norm(x)^2)/L;
disp(['Power of the Signal from Time domain ',num2str(P)]);

L=length(x); NFFT=L;
X=fftshift(fft(x,NFFT));          
Px=X.*conj(X)/(L^2); %Power of each freq components          
fVals=fs*(-NFFT/2:NFFT/2-1)/NFFT;       
subplot(1,2,2); stem(fVals,Px,'b'); title('Power Spectral Density');
xlabel('Frequency (Hz)');ylabel('Power');