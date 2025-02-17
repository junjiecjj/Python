function [SIGNAL,fVals]=freqDomainView(signal,Fs,type)
%Returns raw FFT values & frequency bins for the time domain signal
%signal - discrete-time domain representation of a signal
%Fs - sampling frequency of the discrete-time representation
%type - 'single' or 'double' - returns the single/double sided FFT
NFFT=2^nextpow2(length(signal)); %FFT length
if (nargin ==1), Fs=1; type='double'; end
if (nargin==2), type='double'; end
if strcmpi(type,'single') %single sided FFT
    SIGNAL=fft(signal,NFFT); 
    SIGNAL=SIGNAL(1:NFFT/2); %Throw away half the number of values
    fVals=Fs*(0:NFFT/2-1)/NFFT;
else %double sided FFT
    SIGNAL=fftshift(fft(signal,NFFT)); 
    fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
end,end