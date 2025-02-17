function [X,ref]=modulation_mapper(MOD_TYPE,M,d)
%Modulation mapper for OFDM transmitter
%  MOD_TYPE - 'MPSK' or 'MQAM' modulation
%  M - modulation order, For BPSK M=2, QPSK M=4, 256-QAM M=256 etc..,
%  d - data symbols to be modulated drawn from the set {1,2,...,M}
%returns
%  X - modulated symbols 
%  ref -ideal constellation points that could be used by IQ detector
if strcmpi(MOD_TYPE,'MPSK'),
    [X,ref]=mpsk_modulator(M,d);%MPSK modulation
else
    if strcmpi(MOD_TYPE,'MQAM'),
        [X,ref]=mqam_modulator(M,d);%MQAM modulation
    else
        error('Invalid Modulation specified');
    end
end;end