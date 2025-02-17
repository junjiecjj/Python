%Demonstrate extraction of instantaneous amplitude and phase from 
%the analytic signal constructed from a real-valued modulated signal.
fs = 600; %sampling frequency in Hz
t = 0:1/fs:1-1/fs; %time base
a_t = 1.0 + 0.7 * sin(2.0*pi*3.0*t) ; %information signal
c_t = chirp(t,20,t(end),80); %chirp carrier
x = a_t .* c_t; %modulated signal
subplot(2,1,1); plot(x);hold on; %plot the modulated signal

z = analytic_signal(x); %form the analytical signal
inst_amplitude = abs(z); %envelope extraction
inst_phase = unwrap(angle(z));%inst phase
inst_freq = diff(inst_phase)/(2*pi)*fs;%inst frequency

%Regenerate the carrier from the instantaneous phase
regenerated_carrier = cos(inst_phase);

plot(inst_amplitude,'r'); %overlay the extracted envelope
title('Modulated signal & extracted envelope');xlabel('n');ylabel('x(t) & |z(t)|');
subplot(2,1,2); plot(cos(inst_phase));
title('Extracted carrier or TFS');xlabel('n'); ylabel('cos[\omega(t)]');