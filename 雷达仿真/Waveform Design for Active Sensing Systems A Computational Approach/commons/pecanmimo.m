function X = pecanmimo(N, M, X0)
% X = pecanmimo(N, M) or pecanmimo(N, M, X0), PeCAN MIMO
%   N: length of each transmit sequence
%   M: number of transmit sequences
%   X0: N-by-M, initialization sequence set

if nargin == 3
    X = X0;
else
    X = exp(1i * 2*pi * rand(N, M));
end

XPrev = zeros(N, M);
iterDiff = norm(X - XPrev, 'fro');

while (iterDiff > 1e-5)
    %disp(iterDiff);
    XPrev = X;
    % fix X
    fftX = 1/sqrt(N) * fft(X); % N-by-M
    V = zeros(N,M);
    for n = 1:N
        V(n,:) = fftX(n,:) / norm(fftX(n,:));
    end
    % fix V
    ifftV = sqrt(N) * ifft(V); % N-by-M
    X = exp(1i * angle(ifftV));
    
%     % quantization
%     phi = angle(X);
%     phi(phi<0) = phi(phi<0) + 2*pi;
%     X = exp(1i * round(phi/(2*pi/16))*(2*pi/16));
    
    % stop criterion
    iterDiff = norm(X - XPrev, 'fro');
end
