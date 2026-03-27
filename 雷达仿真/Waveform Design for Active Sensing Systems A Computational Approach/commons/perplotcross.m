function [r PSL] = perplotcross(x1, x2, name)
% [r PSL] = perplotcross(x1, x2) or perplotcross(x1, x2, 'name')
%   x1, x2: N-by-1 unimodular sequences
%   name: a string, if nonempty, a correlation plot is shown with the name
%   r: periodic cross-correlation of x1 & x2.
%   PSL: peak sidelobe level

N = length(x1);
r = zeros(2*(N-1)+1, 1);

for k = (-N+1):(N-1)
    x2Shift = circshift(x2, k);
    r(k+N) = x2Shift' * x1;
end

if nargin == 3 % plot the correlation level 20*log10(r(k)/r(0))
    % figure;
    plot(-(N-1):(N-1), 20*log10(abs(r)/N)); hold on;
    xlabel('k'); ylabel('$|\tilde{r}(k)|/N$ (dB)', 'Interpreter', 'LaTex');
    title(name);
    V = axis;
    axis([-N+1 N-1 V(3) 0]);
    myboldify;
    drawnow;
end

if nargout == 2
    PSL = max(abs(r));
end