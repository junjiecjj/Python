function [r PSL] = aperplotcross(x1, x2, name, y_range)
% [r PSL] = aperplotcross(x1, x2) or aperplotcross(x1, x2, 'name')
%   x1, x2: N-by-1 unimodular sequences
%   name: a string, if nonempty, a correlation plot is shown with the name
%   r: aperiodic cross-correlation of x1 & x2.
%   PSL: peak sidelobe level

N = length(x1);
r = zeros(2*(N-1)+1, 1);

for k = 0:(N-1)
    x2Shift = zeros(N, 1);
    x2Shift((k+1):end) = x2(1:(N-k));
    r(k+N) = x2Shift' * x1;
end
for k = (-N+1):(-1)
    x2Shift = zeros(N, 1);
    x2Shift(1:(N+k)) = x2((1-k):N);
    r(k+N) = x2Shift' * x1;
end

if nargin >= 3 % plot the correlation level 20*log10(r(k)/r(0))
    figure;
    plot(-(N-1):(N-1), 20*log10(abs(r)/N));
    xlabel('k'); ylabel('|r(k)|/N (dB)');
    title(name);
    if nargin == 3
        V = axis;
        axis([-N+1 N-1 V(3) 0]);
    else
        axis([-N+1 N-1 y_range(1) y_range(2)]);
    end
    myboldify;
    drawnow;
end

if nargout == 2
    PSL = max(abs(r));
end