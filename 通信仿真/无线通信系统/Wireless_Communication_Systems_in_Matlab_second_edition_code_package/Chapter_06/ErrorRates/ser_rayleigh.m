function [ser] = ser_rayleigh(EbN0dB,MOD_TYPE,M)
%Compute Theoretical Symbol Error rates for MPSK or MQAM modulations
%EbN0dB - list of SNR per bit points
%MOD_TYPE - 'MPSK' or 'MQAM'
%M - Modulation level for the chosen modulation
%  - For MPSK M can be any power of 2
%  - For MQAM M must be even power of 2 (square QAM only)
gamma_b = 10.^(EbN0dB/10); %SNR per bit in linear scale
gamma_s = log2(M)*gamma_b; %SNR per symbol in linear scale

switch lower(MOD_TYPE)
    case {'bpsk'}
        ser = 0.5*(1-sqrt(gamma_b/(1+gamma_b)));        
    case {'mpsk','psk'}
        ser = zeros(size(gamma_s));
        for i=1:length(gamma_s), %for each SNR point
            g = sin(pi/M).^2;
            fun = @(x) 1./(1+(g.*gamma_s(i)./(sin(x).^2))); %MGF
            ser(i) = (1/pi)*integral(fun,0,pi*(M-1)/M); 
        end        
    case {'mqam','qam'}
        ser = zeros(size(gamma_s));
        for i=1:length(gamma_s), %for each SNR point
            g = 1.5/(M-1);
            fun = @(x) 1./(1+(g.*gamma_s(i)./(sin(x).^2)));%MGF
            ser(i) = 4/pi*(1-1/sqrt(M))*integral(fun,0,pi/2)-4/pi*(1-1/sqrt(M))^2*integral(fun,0,pi/4);
        end        
    case {'mpam','pam'}
        ser = zeros(size(gamma_s));
        for i=1:length(gamma_s), %for each SNR point
            g = 3/(M^2-1);
            fun = @(x) 1./(1+(g.*gamma_s(i)./(sin(x).^2)));%MGF
            ser(i) = 2*(M-1)/(M*pi)*integral(fun,0,pi/2);
        end        
end
end