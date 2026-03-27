function x = cansisostoppar(rho, N, NFull, stopIndex, lambda, x0)
% x = cansisostoppar(rho, N, NFull, stopIndex, lambda, x0), Stopband SISO CAN with
% PAR constraint
%   rho: PAR(x) = max{|x(n)|^2} <= rho (average power is 1)
%   N: the length of the sequence x
%   NFull: the length of x after padding zeros in the tail
%   stopIndex: an NFull-by-1 binary vector corresponding to NFull FFT
%       frequencies. 1 corresponds to a frequency that should be avoided
%   lambda: 0<lambda<1 controls relative weighting: lambda for stopband
%       suppression and 1-lambda for range sidelobe suppression
%   x0: N-by-1, the initialization sequence
%   x: N-by-1, the generated sequence

if nargin == 6
    x = x0;
else
    x = exp(1i * 2*pi * rand(N,1));
end

stopIndex = logical(stopIndex);

xPre = zeros(N,1);
iterDiff = norm(x - xPre);

while (iterDiff > 1e-3)
    disp(['iterDiff = ' num2str(iterDiff)]);
    xPre = x;
    % minimize w.r.t v
    v = 1/sqrt(2) * exp(1i * angle(fft(x, 2*N))); % 2N-by-1
    % minimize w.r.t x
    w = 1/sqrt(NFull) * fft(x, NFull); % NFull-by-1
    w(stopIndex) = 0;
    c1 = sqrt(NFull) * ifft(w); % NFull-by-1
    
    c2 = sqrt(2*N) * ifft(v); % 2N-by-1
    xGoal = lambda * c1(1:N) + (1-lambda) * c2(1:N); % N-by-1
    if rho == 1
        x = exp(1i * angle(xGoal));
    else
        x = vectorfitpar(xGoal, N, rho); % N-by-1
    end
    % stop criterion
    iterDiff = norm(x - xPre);
end