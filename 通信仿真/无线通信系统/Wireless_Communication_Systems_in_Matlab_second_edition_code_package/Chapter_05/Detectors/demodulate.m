function [dCap]=demodulate(MOD_TYPE,M,r,COHERENCE)
%Wrapper functin to call various digital demodulation techniques
%  MOD_TYPE - 'PSK','QAM','PAM','FSK'
%  M - modulation order, For BPSK M=2, QPSK M=4, 256-QAM M=256 etc..,
% COHERENCE - only applicable if FSK modulation is chosen
%           - 'coherent' for coherent MFSK
%           - 'noncoherent' for coherent MFSK
%  r - received modulated symbols
%  dCap - demodulated information symbols

%Construct the reference constellation for the selected mod. type
switch lower(MOD_TYPE)
    case 'bpsk'
        dCap= mpsk_detector(2,r); %M=2
    case 'psk'
        dCap= mpsk_detector(M,r);        
    case 'qam'
        dCap= mqam_detector(M,r);        
    case 'pam'
        dCap= mpam_detector(M,r);
    case 'fsk'
        dCap= mfsk_detector(M,r,COHERENCE);
    otherwise
        error('Invalid Modulation specified');
end
