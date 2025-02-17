Tsym=1; %Symbol duration in seconds
L=16; % oversampling rate, each symbol contains L samples
Nsym = 80; %filter span in symbol duration
Fs=L/Tsym;%sampling frequency

[p,t]=rectFunction(L,Nsym); %Rectangular Pulse
subplot(1,2,1);
t=Tsym*t; plot(t,p,'LineWidth',1.5); 
[fftVals,freqVals]=freqDomainView(p,Fs,'double'); %See Chapter 1
subplot(1,2,2);
plot(freqVals,abs(fftVals)/abs(fftVals(length(fftVals)/2+1)));