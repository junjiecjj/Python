function show_criterion(s, Gamma, beta, name, w)
% show_criterion(s, Gamma), compute the correlation, spectrum and MSE
%	s: N-by-1, the designed probing sequence
%	Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%	name: e.g., 'CAN', 'PRE', 'CYC', 'TIM', 'FRE'

N = length(s);
R = Rcalc(s, Gamma, beta);
if nargin < 5
    w = R \ s; % the receive filter
end
mse = real(w' * R * w) / (abs(w'*s))^2; % 1/(s'*inv(R)*s)

r = xcorr(w, s) / abs(w'*s); % (2N-1)-by-1
figure;
plot(-(N-1):(N-1), 20*log10(abs(r)), 'b');
xlabel('k'); ylabel('|r_{ws}(k)|/N (dB)');
title(name);
axis([-(N-1) N-1 -80 0]);
myboldify;

ssptrm = abs(fft(s,800));
ssptrm = ssptrm / abs(ssptrm);
figure;
plot(linspace(0,1,length(ssptrm)), 20*log10(ssptrm), 'b');
xlabel('Frequency (Hz)');
ylabel('Power Spectrum (dB)');
title(name);
ylim([-80 0]);
myboldify;

w = w/norm(w) * sqrt(N);
wsptrm = abs(fft(w,800));
wsptrm = wsptrm / abs(wsptrm);
figure;
plot(linspace(0,1,length(wsptrm)), 20*log10(wsptrm), 'b');
xlabel('Frequency (Hz)');
ylabel('Frequency Response (dB)');
title(name);
ylim([-80 0]);
myboldify;

rtmp = abs(r);
rtmp(N) = 0;
disp(name);
disp(['    MSE: ' num2str(10*log10(mse)) ' dB    (' num2str(mse) ')']);
disp(['    ISL: ' num2str(10*log10(sum(rtmp.^2))) ' dB']);
disp(['    PSL: ' num2str(20*log10(max(rtmp))) ' dB']);
