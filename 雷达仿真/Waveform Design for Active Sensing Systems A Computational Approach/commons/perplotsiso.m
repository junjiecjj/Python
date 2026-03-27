function [r ISL PSL] = perplotsiso(x, name)
% [r ISL PSL] = perplotsiso(x) or perplotsiso(x, name), periodic 
% correlation plot of a single sequence x
%   x: N-by-1, a unimodular sequence
%   name: a string, if nonempty, a correlation plot is shown with the name
%   as its title
%   r: (2N-1)-by-1, the periodic correlation level of x
%   ISL: integrated sidelobe level, \sum_{k=1,...,N-1} |r(k)|^2
%   PSL: peak sidelobe level, max |r(k)|

N = length(x);
r = zeros(2*(N-1)+1, 1);

for k = 0:(N-1)
    xShift = circshift(x, k);
    r(k+N) = xShift' * x;
end
r(1:(N-1)) = conj(r(end:-1:(N+1)));

if nargin == 2 % plot the correlation level 20*log10(r(k)/r(0))
    figure;
    plot(-(N-1):(N-1),  20*log10(abs(r)/abs(r(N)))); hold on;
    xlabel('lag'); ylabel('Auto Correlation (dB)');
    title(name);
    V = axis;
    axis([-N+1 N-1 V(3) 0]);
    myboldify;
    drawnow;
end

if nargout >= 2
    ISL = sum((abs(r(1:N-1))).^2);
    if nargout == 3
        PSL = max(abs(r(1:N-1)));
    end
end