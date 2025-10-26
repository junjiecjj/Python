
clc
clear all
close all 
waveform = phased.RectangularWaveform;
x = waveform();
PRF = waveform.PRF;
[afmag,delay,doppler] = ambgfun(x,waveform.SampleRate, PRF);
contour(delay,doppler,afmag);
xlabel("Delay (seconds)");
ylabel("Doppler Shift (hertz)");

size(x)  % 100, 1
waveform.SampleRate % 1000000
PRF   % 10000

size(afmag)  %   256 199 

size(delay)   % 1 199

size(doppler)   % 1 256