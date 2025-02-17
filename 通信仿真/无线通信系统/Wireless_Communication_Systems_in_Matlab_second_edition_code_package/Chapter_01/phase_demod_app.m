%Demonstrate simple Phase Demodulation using Hilbert transform
clearvars; clc;
fc = 210; %carrier frequency
fm = 10; %frequency of modulating signal
alpha = 1; %amplitude of modulating signal
theta = pi/4; %phase offset of modulating signal
beta = pi/5; %constant carrier phase offset 
receiverKnowsCarrier= 'False';
%Set True if receiver knows carrier frequency & phase offset

fs = 8*fc; %sampling frequency
duration = 0.5; %duration of the signal
t = 0:1/fs:duration-1/fs; %time base
%Phase Modulation
m_t = alpha*sin(2*pi*fm*t + theta); %modulating signal
x = cos(2*pi*fc*t + beta + m_t ); %modulated signal

figure(); subplot(2,1,1); plot(t,m_t) %plot modulating signal
title('Modulating signal'); xlabel('t'); ylabel('m(t)')
subplot(2,1,2); plot(t,x) %plot modulated signal
title('Modulated signal'); xlabel('t');ylabel('x(t)')

%Add AWGN noise to the transmitted signal
nMean = 0; nSigma = 0.1; %noise mean and sigma
n = nMean + nSigma*randn(size(t)); %awgn noise
r = x + n;  %noisy received signal

%Demodulation of the noisy Phase Modulated signal
z= hilbert(r); %form the analytical signal from the received vector
inst_phase = unwrap(angle(z)); %instaneous phase

%If receiver knows the carrier freq/phase perfectly
if strcmpi(receiverKnowsCarrier,'True')
    offsetTerm = 2*pi*fc*t+beta; 
else %else, estimate the subtraction term
    p = polyfit(t,inst_phase,1);%linearly fit the instaneous phase
    %re-evaluate the offset term using the fitted values
    estimated = polyval(p,t); offsetTerm = estimated;
end    
demodulated = inst_phase - offsetTerm;
figure(); plot(t,demodulated); %demodulated signal
title('Demodulated signal'); xlabel('n'); ylabel('\hat{m(t)}');