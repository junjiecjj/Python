function [PatOrgl Theta fSet] = freSpaPattern(X, fc, B, fs, K, ...
    figname2D, figname3D)
% freSpaPattern: draw the spectral&spatial beampattern of X
% PatOrgl = freSpaPattern(X, fc, B, K)
%   PatOrgl: N-by-K (N is # of freq bins; K is # of spatial grid points)
%   X: N-by-M (M sequences each of which is of length N)
%   fc: carrier frequency
%   B: bandwidth of the signal
%   fs: sampling frequency

[N M] = size(X);
fSet = linspace((-N/2)/N * fs, (N/2-1)/N * fs, N); % N frequency bins
c = 3*10^8; % speed of light
d = 1/2 * (c/(fc+B/2)); % inter-element spacing
Theta = linspace(0,180,K)' * pi/180; % spatial grid

A = zeros(K,M,N); % A(:,:,p) = A_p in the paper
for p = 1:N
    fp = fSet(p);
    A(:,:,p) = exp(-1i * 2*pi * cos(Theta) * (fp + fc) * (0:(M-1))*d / c);
end
S = (fft(X,N)).'; % M-by-N
for m = 1:M
    S(m,:) = fftshift(S(m,:));
end

Pat = zeros(K,N);
for p = 1:N
    Pat(:,p) = A(:,:,p) * S(:,p); % K-by-1
end
Pat = Pat.'; % N-by-K
PatOrgl = abs(Pat); % non-normalized pattern
Pat = (abs(Pat)).^2;

%trim the pattern to within [-B/2,B/2]
[tmp indexLeft] = min(abs(fSet+B/2));
[tmp indexRight] = min(abs(fSet-B/2));
fSet = fSet(indexLeft:indexRight);
Pat = Pat(indexLeft:indexRight, :);

Pat = Pat / N;
figure;
imagesc(Theta*180/pi, (fc + fSet)/(1e9), 10*log10(Pat), ...
    [-40 10*log10(max(Pat(:)))]);
set(gca, 'YDir', 'normal');
colormap('hot');
colorbar;
xlabel('Angle (degree)');
ylabel('Frequency (GHz)');
myboldify;
if nargin == 7
    myresize(figname2D);
end

% we cut Pat from -40 dB
Pat(10*log10(Pat)<-40) = 1e-4;
figure;
mesh(Theta*180/pi, (fc + fSet)/(1e9), 10*log10(Pat));
%colormap(flipud(colormap('hot')));
%colorbar;
axis([Theta(1)*180/pi Theta(end)*180/pi ...
    (fc+fSet(1))/(1e9) (fc+fSet(end))/(1e9) -40 10*log10(max(Pat(:)))]);
xlabel('Angle (degree)');
ylabel('Frequency (GHz)');
myboldify;
if nargin == 7
    myresize(figname3D);
end