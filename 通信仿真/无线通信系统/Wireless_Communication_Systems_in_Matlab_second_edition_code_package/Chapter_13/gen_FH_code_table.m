function freqTable = gen_FH_code_table(Fbase,Fspace,Fs,Lh,N)
%Generate frequency translation table for Frequency Hopping (FH)
%Fbase - base frequency (Hz) of the hop
%Fspace - channel spacing (Hz) of the hop
%Fs - sampling frequency (Hz)
%Lh - num of discrete time samples in each hop period
%N - num of total hopping frequencies required(full period of LFSR)
%Return the frequency translation table
t = (0:1:Lh-1)/Fs; %time base for each hopping period
freqTable = zeros(N,Lh);%Table to store N different freq waveforms
for i=1:N, %generate frequency translation table for all N states
    Fi=Fbase+(i-1)*Fspace;
    freqTable(i,:) = cos(2*pi*Fi*t);
end