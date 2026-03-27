function [Pat Theta fSet] = phaseArrayPattern(N, M, fc, B, K, theta0)
% freSpaPattern: draw the spectral&spatial beampattern of X
% Pat = freSpaPattern(X, fc, B, K)
%   Pat: N-by-K (N is # of freq bins; K is # of spatial grid points)
%   M: number of array elements
%   fc: carrier frequency
%   B: bandwidth of the signal
%   theta0: the pointing angle, in degree

fSet = linspace(-B/2, B/2, N);
Theta = linspace(0,180,K)' * pi/180; % spatial grid
c = 3e8; % speed of light
d = 1/2 * (c/(fc+B/2)); % inter-element spacing
theta0 = theta0 * pi/180;

Pat = zeros(N,K);
for n = 1:N
    f = fSet(n);
    for k = 1:K
        theta = Theta(k);
        Pat(n,k) = N * (abs(sum(exp(1i * 2*pi * (f + fc) * ...
            (0:(M-1)) * d * (cos(theta0) - cos(theta))/c))))^2;
    end
end

%Pat = Pat / max(abs(Pat(:)));
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
