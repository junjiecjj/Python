%***************Source*********************
nBits = 60;%number of source bits to transmit
Rb = 20e3; %bit rate of source information in bps

%**********BFSK definitions*****************
fsk_type = 'NONCOHERENT'; %BFSK generation type at the transmitter
h = 1; %modulation index (0.5=coherent BFSK/1= non-coherent BFSK)
Fc = 50e3; %center frequency of BFSK

%**********Frequency Allocation*************
Fbase = 200e3; %The center frequency of the first channel
Fspace = 100e3;%freq. separation between adjacent hopping channels
Fs = 10e6; %sufficiently high sampling frequency for discretization

%*********Frequency Hopper definition*******
G = [1 0 0 1 1]; X=[0 0 0 1];%LFSR generator poly and seed
hopType = 'FAST_HOP'; %FAST_HOP or SLOW_HOP for frequency hopping

%--------Derived Parameters-------
Tb = 1/Rb ;%bit duration of each information bit.
L = Tb*Fs;%num of discrete-time samples in each bit
Fd = h/Tb; %frequency separation of BFSK frequencies

%Adjust num of samples in a hop duration based on hop type selected
if strcmp(hopType,'FAST_HOP'),%hop duration less than bit duration
    Lh = L/4; %4 hops in a bit period
    nHops = 4*nBits; %total number of Hops during the transmission
else%default set to SLOW_HOP: hop duration more than bit duration
    Lh = L*4; %4 bit worth of samples in a hop period
    nHops = nBits/4; %total number of Hops during the transmission
end

%-----Simulate the individual blocks----------------------
d = rand(1,nBits) > 0.5 ; %random information bits
[s_m,t,phase,dt]=bfsk_mod(d,Fc,Fd,L,Fs,fsk_type);%BFSK modulation

c = hopping_chip_waveform(G,X,nHops,Lh,Fbase,Fspace,Fs);%Hopping wfm
s = s_m.*c.';%mix BFSK waveform with hopping frequency waveform

n = 0;%Left to the reader -modify for AWGN noise(see prev chapters)
r = s+n; %received signal with noise
v = r.*c.'; %mix received signal with synchronized hopping freq wave

d_cap = bfsk_noncoherent_demod(v,Fc,Fd,L,Fs); %BFSK demod
bie = sum(d ~= d_cap); %Calculate bits in error
disp(['Bits in Error: ', num2str(bie)]);

%--------Plot waveforms at various stages of tx/rx-------
figure;
subplot(2,1,1);plot(t,dt) ;title('Source bits - d(t)'); 
subplot(2,1,2);plot(t,s_m);title('BFSK modulated - s_m(t)')

figure;
subplot(3,1,1);plot(t,s_m);title('BFSK modulated - s_m(t)')
subplot(3,1,2);plot(t,c);title('Hopping waveform at Tx - c(t)')
subplot(3,1,3);plot(t,s);title('FHSS signal - s(t)')

figure;
subplot(3,1,1);plot(t,r);title('Received signal - r(t)')
subplot(3,1,2);plot(t,c);title('Synced hopping waveform at Rx-c(t)')
subplot(3,1,3);plot(t,v);title('Signal after mixing with hop pattern-v(t)')