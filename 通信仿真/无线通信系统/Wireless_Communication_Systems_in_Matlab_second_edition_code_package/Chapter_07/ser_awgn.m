function [SER] = ser_awgn(EbN0dB,MOD_TYPE,M,COHERENCE)
%Theoretical Symbol Error Rate for various modulations over AWGN
%EbN0dB - list of SNR per bit values
%MOD_TYPE - 'BPSK','PSK','QAM','PAM','FSK'
%M - Modulation level for the chosen modulation
%  - For PSK,PAM,FSK M can be any power of 2
%  - For QAM M must be even power of 2 (square QAM only)
%Parameter COHERENCE is only applicable for FSK modulation
%COHERENCE = 'coherent' for coherent FSK detection
%          = 'noncoherent' for noncoherent FSK detection
gamma_b = 10.^(EbN0dB/10); %SNR per bit in linear scale
gamma_s = log2(M)*gamma_b; %SNR per symbol in linear scale
SER = zeros(size(EbN0dB));

switch lower(MOD_TYPE)
    case 'bpsk'
        SER=0.5*erfc(sqrt(gamma_b));
    case {'psk','mpsk'}
        if M==2, %for BPSK
            SER=0.5*erfc(sqrt(gamma_b));
        else
            if M==4, %for QPSK
                Q=0.5*erfc(sqrt(gamma_b)); SER=2*Q-Q.^2;
            else %for other higher order M-ary PSK
                SER=erfc(sqrt(gamma_s)*sin(pi/M));
            end
        end
    case {'qam','mqam'}
        SER = 1-(1-(1-1/sqrt(M))*erfc(sqrt(3/2*gamma_s/(M-1)))).^2;
    case {'fsk','mfsk'}
        if strcmpi(COHERENCE,'coherent'),
            for ii=1:length(gamma_s),
                fun = @(q) (0.5*erfc((-q - sqrt(2.*gamma_s(ii)))/sqrt(2))).^(M-1).* 1/sqrt(2*pi).*exp(-q.^2/2);
                SER(ii) = 1-integral(fun,-inf,inf);
            end
        else %Default compute for noncoherent
            for jj=1:length(gamma_s),
                summ=0;
                for i=1:M-1,
                    n=M-1; r=i; %for nCr formula
                    summ=summ+(-1).^(i+1)./(i+1).*prod((n-r+1:n)./(1:r)).*exp(-i./(i+1).*gamma_s(jj));
                end
                SER(jj)=summ; %Theoretical SER for non-coherent detection
            end
        end
    case {'pam','mpam'}
        SER=2*(1-1/M)*0.5*erfc(sqrt(3*gamma_s/(M^2-1)));
    otherwise
        display 'ser_awgn.m: Invalid modulation (MOD_TYPE) selected'            
end
end