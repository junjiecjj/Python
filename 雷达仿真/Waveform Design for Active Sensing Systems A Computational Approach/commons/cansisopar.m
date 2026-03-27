function x = cansisopar(rho, N, x0)
% x = cansiso(rho, N) or x = cansiso(rho, N, x0), CAN algorithm with the
% PAR (peak-to-average ratio) constraint
%   rho: PAR(x) = max{|x(n)|^2} <= rho (average power is 1)
%   N: length of the sequence
%   x0: N-by-1, the initialization sequence
%   x: N-by-1, the generated sequence

if nargin == 3
    x = x0;
else
    x = exp(1i * 2*pi * rand(N,1));
end

xPre = zeros(N, 1);
iterDiff = norm(x - xPre);

while (iterDiff>1e-3)
    %disp(iterDiff);
    xPre = x;
    % step 2
    z = [x; zeros(N, 1)]; % 2N-by-1
    f = 1/sqrt(2*N) * fft(z); % 2N-by-1
    v = sqrt(1/2) * exp(1i * angle(f)); % 2N-by-1
    % step 1
    g = sqrt(2*N) * ifft(v); % 2N-by-1    
    x = vectorfitpar(g(1:N), N, rho); % N-by-1
    % stop criterion
    iterDiff = norm(x - xPre);
end