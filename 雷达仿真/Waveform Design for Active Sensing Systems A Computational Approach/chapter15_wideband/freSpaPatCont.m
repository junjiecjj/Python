function [Pat Theta fSet] = freSpaPatCont(X, fc, B, fs, K, fd, fPlot, ...
    fignameSpectrum, figname2D, figname3D)
% freSpaPattern: draw the spectral&spatial beampattern of the CONTINUOUS X
% Pat = freSpaPattern(X, fc, B, K)
%   Pat: N-by-K (N is # of freq bins; K is # of spatial grid points)
%   X: N-by-M (M sequences each of which is of length N)
%   fc: carrier frequency
%   B: bandwidth of the signal
%   fs: sampling frequency
%   fd: the sampling frequency after raised-cosine (fd must be an integer
%   multiple of fs)
%   fPlot: [fPlot(1) fPlot(2)] specifies the (baseband) frequency plot range

[N M] = size(X);

% pass the sequence through a raised-cosine filter
if (mod(fd, fs) ~= 0)
    error('fd must be an integer multiple of fs');
end
rolloff = 0.5;
delay = 3;
Xbar = rcosflt(X, fs, fd, 'fir/normal', rolloff, delay); % raised-cosine filtering
% Xbar has M columns
% the length of each column is L = (N + 2 delay) fd/fs where delay=3
L = size(Xbar, 1); % length of the filtered sequence
% normalize Xbar so that the energy of each column is N
for m = 1:M
    Xbar(:,m) = Xbar(:,m) * sqrt(N)/norm(Xbar(:,m));
end

% spectrum of the real continuous baseband signal
S = (fft(Xbar,L)).'; % M-by-L
for m = 1:M
    S(m,:) = fftshift(S(m,:));
end
figure;
for m = 1:M
    SD = abs(S(m,:)).^2/L^2; % spectral density
    SD = SD / max(SD);
    plot(((-L/2):(L/2-1))*(fd/L)/(1e9), 10*log10(SD)); hold on;
end
plot([-B/2/1e9 -B/2/1e9], [-100 0], 'r--'); hold on;
plot([B/2/1e9 B/2/1e9], [-100 0], 'r--'); hold off;
xlabel('frequency (GHz)');
ylabel('Spectral Density of the M Baseband Signals (dB)');
axis([(-2*N)*(fd/L)/1e9 (2*N)*(fd/L)/1e9 -100 0]);
myboldify;
if nargin == 10
    myresize(fignameSpectrum);
end


% % keep only the frequency within [-fs/2,fs/2]
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %            before RaisedCosine  |  after
% % length            N                      L = fd/fs (N+2 delay)
% % f_samp          fs                      fd
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % It is easy to see that fs/2 corresponds to the DFT grid point at position
% % (N/2 + delay), and -fs/2 corresponds to the position -(N/2 + delay)
% indexTemp = round(-(N/2 + delay) - (-L/2) + 1);
% Nbar = N + 2 * delay;
% S = S(:, indexTemp:indexTemp+Nbar-1); % M-by-Nbar
% fSet = linspace((-N/2)/N * fs, (N/2-1)/N * fs, Nbar); 

% keep the frequencies within [fPlot(1) fPlot(2)]
indexTemp = round((fPlot - (-fd/2))/(fd/L));
S = S(:, indexTemp(1):indexTemp(2));
Nbar = indexTemp(2) - indexTemp(1); % No. of frequency bins
fSet = linspace(fPlot(1), fPlot(2), Nbar);

c = 3*10^8; % speed of light
d = 1/2 * (c/(fc+B/2)); % inter-element spacing
Theta = linspace(0,180,K)' * pi/180; % spatial grid
A = zeros(K,M,Nbar); % A(:,:,p) = A_p in the paper
for p = 1:Nbar
    fp = fSet(p);
    A(:,:,p) = exp(-1i * 2*pi * cos(Theta) * (fp + fc) * (0:(M-1))*d / c);
end

Pat = zeros(K,Nbar);
for p = 1:Nbar
    Pat(:,p) = A(:,:,p) * S(:,p); % K-by-1
end
Pat = Pat.'; % Nbar-by-K
Pat = (abs(Pat)).^2;

% \sum |y(p)|^2 = LN
% Note that the energy is concentrated within the middle Nbar frequency
% bins, so a better normalization is given by LN/Nbar, instead of LN/L=N
Pat = Pat / (L * N / Nbar);
figure;
imagesc(Theta*180/pi, (fc + fSet)/1e9, 10*log10(Pat), ...
    [-40 10*log10(max(Pat(:)))]);
set(gca, 'YDir', 'normal');
colormap('hot');
colorbar;
xlabel('Angle (degree)');
ylabel('Frequency (GHz)');
myboldify;
if nargin == 10
    myresize(figname2D);
end

% we cut Pat from -40 dB
Pat(10*log10(Pat)<-40) = 1e-4;
figure;
mesh(Theta*180/pi, (fc + fSet)/1e9, 10*log10(Pat));
axis([Theta(1)*180/pi Theta(end)*180/pi ...
    (fc+fSet(1))/1e9 (fc+fSet(end))/1e9 -40 10*log10(max(Pat(:)))]);
xlabel('Angle (degree)');
ylabel('Frequency (GHz)');
myboldify;
if nargin == 10
    myresize(figname3D);
end