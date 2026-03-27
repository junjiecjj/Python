function [r ISL PSL] = aperplotsiso(x, name, y_range)
% [r ISL PSL] = aperplotsiso(x), aperplotsiso(x, name) or 
% aperplotsiso(x, name, y_range), aperiodic correlation plot of a single sequence x
%   x: N-by-1, a unimodular sequence
%   name: a string, if nonempty, a correlation plot is shown with the name
%       as its title
%   y_range: [ymin ymax], the y-axis limit for plot
%   r: (2N-1)-by-1, the aperiodic correlations of x
%   ISL: integrated sidelobe level, \sum_{k=-N+1,...,-1,1,...,N-1} |r(k)|^2
%   PSL: peak sidelobe level, max |r(k)|

N = length(x);
r = zeros(2*(N-1)+1, 1);

for k = 0:(N-1)
    xShift = zeros(N, 1);
    xShift((k+1):end) = x(1:(N-k));
    r(k+N) = xShift' * x;
end
r(1:(N-1)) = conj(r(end:-1:(N+1)));

if nargin >= 2 % plot the correlation level 20*log10(r(k)/r(0))
    figure;
    plot(-(N-1):(N-1), 20*log10(abs(r)/N));
    xlabel('k'); ylabel('|r(k)|/N (dB)');
    title(name);
    if nargin == 2
        V = axis;
        axis([-N+1 N-1 V(3) 0]);
    else
        axis([-N+1 N-1 y_range(1) y_range(2)]);
    end
    myboldify;
    drawnow;
end

if nargout >= 2
    ISL = sum((abs(r)).^2) - N^2;
    if nargout == 3
        PSL = max(abs(r(1:N-1)));
    end
end