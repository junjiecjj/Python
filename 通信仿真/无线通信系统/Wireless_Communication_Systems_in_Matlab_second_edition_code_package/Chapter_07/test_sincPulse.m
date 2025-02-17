Tsym=1; %Symbol duration
L=16;  %oversampling rate, each symbol contains L samples
Nsym = 80; %filter span in symbol duration
Fs=L/Tsym; %sampling frequency

[p,t]=sincFunction(L,Nsym); %Sinc Pulse
subplot(1,2,1); t=t*Tsym; plot(t,p); title('Sinc pulse');    
[fftVals,freqVals]=freqDomainView(p,Fs,'double'); %See Chapter 1
subplot(1,2,2); 
plot(freqVals,abs(fftVals)/abs(fftVals(length(fftVals)/2+1)));