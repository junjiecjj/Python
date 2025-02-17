f=10; %frequency of sine wave
overSampRate=30; %oversampling rate
fs=overSampRate*f; %sampling frequency
phase = 1/3*pi; %desired phase shift in radians
nCyl = 5; %to generate five cycles of sine wave
t=0:1/fs:nCyl*1/f-1/fs; %time base

g=sin(2*pi*f*t+phase); %replace with cos if a cosine wave is desired
plot(t,g); title(['Sine Wave f=', num2str(f), 'Hz']);