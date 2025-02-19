function [hw] = Jakes_filter(f_max,Ts,N)
%FIR channel shaping filter with Jakes doppler spectrum
%S(f) = 1/ [sqrt(1-(f/f_max)^2)*pi*f_max]
%Use impulse response obtained from spectral factorization
%Input parameters are
%f_max = maximum doppler frequency in Hz
%Ts = sampling interval in seconds
%N = desired filter order
%Returns the windowed impulse response of the FIR filter
%Example: fd=10; Ts=0.01; N=512; Jakes_filter(fd,Ts,N)
L=N/2;
n=1:L;
J_pos=besselj(0.25,2*pi*f_max*n*Ts)./(n.^0.25);
J_neg = fliplr(J_pos);
J_0 = 1.468813*(f_max*Ts)^0.25;
J = [J_neg J_0 J_pos]; %Jakes filter
%Shaping filter smoothed using Hamming window
n=0:N; hamm = 0.54-0.46*cos(2*pi*n/N);%Hamming window
hw = J.*hamm;%multiply Jakes filter with the Hamming window
hw = hw./sqrt(sum(abs(hw).^2));%Normalized impulse response

%Plot impulse response and spectrum
figure(1); subplot(1,2,1); plot(hw); axis tight;
title('Windowed impulse response');
xlabel('n'); ylabel('h_w[n]');
%use the function given in section 1.3.4 to plot the spectrum
[fftVals, freqVals] = freqDomainView(hw, 1/Ts, 'double');%section 1.3.4
H_w = (Ts/length(hw))*abs(fftVals).^2;%PSD H(f)
subplot(1,2,2); plot(freqVals,H_w); grid on; xlim([-20 20]);
title('Jakes Spectrum');xlabel('f');ylabel('|H_{w}(f)|^2');
end
