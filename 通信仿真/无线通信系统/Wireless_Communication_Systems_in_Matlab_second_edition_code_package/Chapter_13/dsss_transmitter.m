function [s_t,carrier_ref,prbs_ref]=dsss_transmitter(d,prbsType,G1,...
   G2,X1,X2,Rb,Rc,L)
% Direct Sequence Spread Spectrum (DSSS) transmitter - returns the DSSS
% waveform (s), reference carrier, the prbs reference waveform for use
% in synchronization in the receiver
%d - input binary data stream
%prbsType - type of PRBS generator - 'MSEQUENCE' or 'GOLD'
%  If prbsType == 'MSEQUENCE' G1 is the generator poly for LFSR 
%   and X1 its seed. G2 and X2 are not used
%  If prbsType == 'GOLD' G1,G2 are the generator polynomials 
%   for LFSR1/LFSR2 and X1,X2 are their initial seeds.
%G1,G2 - Generator polynomials for PRBS generation
%X1,X2 - Initial states of LFSRs
%Rb - data rate (bps) for the data d
%Rc - chip-rate (Rc >> Rb AND Rc is integral multiple of Rb) 
%L - oversampling factor for waveform generation

prbs = generatePRBS(prbsType,G1,G2,X1,X2);
prbs=prbs(:); d=d(:); %serialize
dataLen= length(d)*(Rc/Rb);%required PRBS length to cover the data
prbs_ref= repeatSequence(prbs,dataLen);%repeat PRBS to match data

d_t = kron(d,ones(L*Rc/Rb,1)); %data waveform
prbs_t = kron(prbs_ref,ones(L,1)); %spreading sequence waveform
sbb_t = 2*xor(d_t,prbs_t)-1; %XOR data and PRBS, convert to bipolar
n=(0:1:length(sbb_t)-1).'; carrier_ref=cos(2*pi*2*n/L);
s_t = sbb_t.*carrier_ref; %modulation,2 cycles per chip

figure(1); %Plot waveforms
subplot(3,1,1); plot(d_t); title('data sequence'); hold on;
subplot(3,1,2); plot(prbs_t); title('PRBS sequence');
subplot(3,1,3); plot(s_t); title('DS-SS signal (baseband)'); end