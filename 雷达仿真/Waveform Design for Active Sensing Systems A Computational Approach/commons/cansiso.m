function x = cansiso(N, x0)
% x = cansiso(N) or x = cansiso(N, x0), CAN SISO
%   N: length of the sequence
%   x0: N-by-1, the initialization sequence
%   x: N-by-1, the generated sequence

if nargin == 2
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
    x = exp(1i * angle(g(1:N))); % N-by-1
    % x = exp(1i * floor(angle(g(1:N))/(2*pi/L))*(2*pi/L)); % if phase
    % quantization to L levels, e.g. L = 256
    iterDiff = norm(x - xPre);
end