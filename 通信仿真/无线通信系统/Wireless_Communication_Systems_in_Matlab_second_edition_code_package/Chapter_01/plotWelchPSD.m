function [Pxx,F]=plotWelchPSD(SIGNAL,Fs,COLOR)
%Plot PSD of a stochastic SIGNAL using Welch estimate
% SIGNAL - signal vector for which the PSD is plotted
% Fs - Sampling Frequency
% COLOR - color character for the plot
ns = max(size(SIGNAL));
na = 16;%averaging factor to plot averaged welch spectrum
w = hanning(floor(ns/na));%Hanning window
%Welch PSD estimate with Hanning window and no overlap
[Pxx,F]=pwelch(SIGNAL,w,0,[],Fs);
if nargin==3,%color argument is given, plot PSD
  plot(F,10*log10(Pxx),COLOR);%normalize frequency axis
end