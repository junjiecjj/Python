function [X FS] = wican(N, M, dePattern, fc, B, fs, K, rho)
% wican: design a wideband sequence set (N-by-M, M sequences each of which
% is of length N) to fit a spatial&spectral beampattern
%   X = wican(N, M, dePattern, fc, B, K)
%   dePattern: desired beampattern N-by-K
%       (N is # of freq bins; K is # of spatial grid points)
%   fc: carrier frequency
%   B: bandwidth of the signal
%   fs: sampling frequency
%   rho: PAR<=rho (this input parameter can be ommited)
%   X: N-by-M, designed sequence set
%   FS: N-by-M, sequence set with only the total-energy constraint

dePattern = dePattern.'; % K-by-N, to coincide with the paper
if mod(N,2)~=0
    error('N must be even');
end
fSet = linspace((-N/2)/N * fs, (N/2-1)/N * fs, N); % N frequency bins
c = 3*10^8; % speed of light
d = 1/2 * (c/(fc+B/2)); % inter-element spacing
Theta = linspace(0,180,K)' * pi/180; % K spatial grid points

A = zeros(K,M,N); % A(:,:,p) = A_p in the paper
for p = 1:N
    fp = fSet(p);
    A(:,:,p) = exp(-1i * 2*pi * cos(Theta) * (fp + fc) * (0:(M-1))*d / c);
end
S = zeros(M,N); % S(:,p) = s_p in the paper

% Part I, s_p
Phi = 2*pi*rand(K,N);
PhiPre = zeros(K,N);
iterDiff = norm(exp(1i*Phi) - exp(1i*PhiPre), 'fro');
while(iterDiff>1e-3)
    disp(['Stage 1: ' num2str(iterDiff)]);
    PhiPre = Phi;
    % s
    for p = 1:N
        Ap = A(:,:,p); % K-by-M
        bp = dePattern(:,p) .* exp(1i * Phi(:,p)); % K-by-1
        S(:,p) = (Ap' * Ap) \ (Ap' * bp); % M-by-1
    end
    % phi
    for p = 1:N
        Ap = A(:,:,p); % K-by-M
        sp = S(:,p); % M-by-1
        Phi(:,p) = angle(Ap * sp);
    end
    iterDiff = norm(exp(1i*Phi) - exp(1i*PhiPre), 'fro');
end

% Part II: x
Psi = 2*pi * rand(N,1);
PsiPre = zeros(N,1);
iterDiff = norm(exp(1i*Psi(:)) - exp(1i*PsiPre(:)), 'fro');
while (iterDiff > 1e-3)
    disp(['Stage 2: ' num2str(iterDiff)]);
    PsiPre = Psi;
    % fix psi, solve x
    S = S * diag(exp(1i*Psi)); % M-by-N
    FS = zeros(N,M);
    energy = 0;
    for m = 1:M
        FS(:,m) = ifft(fftshift((S(m,:)).'));
        energy = energy + norm(FS(:,m))^2;
    end
    FS = FS * sqrt(N*M / energy);    
    if nargin == 7
        X = exp(1i * angle(FS));
    else
        X = zeros(N, M);
        for m = 1:M
            X(:,m) = vectorfitpar(FS(:,m), N, rho);
        end
    end
    % fix x, solve psi
    FX = fft(X); % N-by-M
    for m = 1:M
        FX(:,m) = fftshift(FX(:,m));
    end
    for n = 1:N
        Psi(n) = angle(FX(n,:) * conj(S(:,n)));
    end
    iterDiff = norm(exp(1i*Psi(:)) - exp(1i*PsiPre(:)), 'fro');
end