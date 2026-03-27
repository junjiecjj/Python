function X = canmimo(N, M, X0)
% X = canmimo(N, M) or canmimo(N, M, X0), CAN MIMO
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
Y = zeros(2*N, M);
V = zeros(2*N, M);

while (iterDiff > 1e-3)
    %disp(iterDiff);
    XPrev = X;
    % step 1
    Y(1:N, :) = X;
    fftY = 1/sqrt(2*N) * fft(Y);
    for k = 1:(2*N)
        V(k,:) = 1/sqrt(2) * fftY(k,:) / norm(fftY(k,:));
    end
    % step 2
    ifftV = sqrt(2*N) * ifft(V);
    X = exp(1i * angle(ifftV(1:N, 1:M)));
    % stop criterion
    iterDiff = norm(X - XPrev, 'fro');
end