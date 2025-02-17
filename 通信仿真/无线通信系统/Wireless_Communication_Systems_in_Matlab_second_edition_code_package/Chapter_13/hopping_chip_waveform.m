function [c] = hopping_chip_waveform(G,X,nHops,Lh,Fbase,Fspace,Fs)
%Generate Frequency Hopping chip sequence based on m-sequence LFSR
%G,X - Generator poly. and initial seed for underlying m-sequence
%nHops - total number of hops needed
%Lh - number of discrete time samples in each hop period
%Fbase - base frequency (Hz) of the hop
%Fspace - channel spacing (Hz) of the hop
%Fs - sampling frequency (Hz)

[prbs,STATES] = lfsr( G, X); %initialize LFSR
N = length(prbs); %PRBS period
LFSRStates= repeatSequence(STATES.',nHops);%repeat LFSR states depending on nHops

freqTable=gen_FH_code_table(Fbase,Fspace,Fs,Lh,N); %freq translation
c = zeros(nHops,Lh); %place holder for the hopping sequence waveform
for i=1:nHops,%for each hop choose one freq wave based on LFSR state
    LFSRState = LFSRStates(i); %given LFSR state
    c(i,:) = freqTable(LFSRState,:);%choose corresponding freq. wave
end
c=c.';c=c(:); %transpose and serialize as a single waveform vector