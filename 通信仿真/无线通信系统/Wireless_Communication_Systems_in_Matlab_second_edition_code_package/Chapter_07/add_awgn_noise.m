function [r,n,N0] = add_awgn_noise(s,SNRdB,L)
%Function to add AWGN to the given signal
%[r,n,N0]= add_awgn_noise(s,SNRdB) adds AWGN noise vector to signal 
%'s' to generate a %resulting signal vector 'r' of specified SNR 
%in dB. It also returns the noise vector 'n' that is added to the
%signal 's' and the spectral density N0 of noise added
%
%[r,n,N0]= add_awgn_noise(s,SNRdB,L) adds AWGN noise vector to 
%signal 's' to generate a resulting signal vector 'r' of specified 
%SNR in dB. The parameter 'L' specifies the oversampling ratio used 
%in the system (for waveform simulation). It also returns the noise 
%vector 'n' that  is added to the signal 's' and the spectral 
%density N0 of noise added	
s_temp=s;
if iscolumn(s), s=s.'; end; %to return the result in same dim as 's' 
gamma = 10^(SNRdB/10); %SNR to linear scale    	
if nargin==2, L=1; end %if third argument is not given, set it to 1

if isvector(s),
    P=L*sum(abs(s).^2)/length(s);%Actual power in the vector
else %for multi-dimensional signals like MFSK
    P=L*sum(sum(abs(s).^2))/length(s); %if s is a matrix [MxN]
end

N0=P/gamma; %Find the noise spectral density
if(isreal(s)),
    n = sqrt(N0/2)*randn(size(s));%computed noise
else
    n = sqrt(N0/2)*(randn(size(s))+1i*randn(size(s)));%computed noise
end 
r = s + n; %received signal
if iscolumn(s_temp), r=r.'; end;%return r in original format as s
end