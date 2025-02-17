f=10; %frequency of sine wave in Hz
overSampRate=30; %oversampling rate
fs=overSampRate*f; %sampling frequency
nCyl = 5; %to generate five cycles of square wave 
t=0:1/fs:nCyl*1/f-1/fs; %time base
g = sign(sin(2*pi*f*t));
plot(t,g); title(['Square Wave f=', num2str(f), 'Hz']);