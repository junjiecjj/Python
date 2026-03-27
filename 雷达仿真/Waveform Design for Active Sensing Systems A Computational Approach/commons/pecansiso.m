function x = pecansiso(N, x0)
% x = pecansiso(N) or x = pecansiso(N, x0), PeCAN SISO
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

while(iterDiff > 1e-10)
    %disp(iterDiff);
    xPre = x;
    % step 1
    f = 1/sqrt(N) * fft(x); % N-by-1
    v = exp(1i * angle(f)); % N-by-1
    % step 2
    g = sqrt(N) * ifft(v); % N-by-1
    x = exp(1i * angle(g)); % N-by-1
    % stop criterion
    iterDiff = norm(x - xPre);
end
