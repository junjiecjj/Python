function [r ISL PSL] = aperplotmimo(X, name, y_range)
% [r ISL PSL] = aperplotmimo(X) or aperplotmimo(X, name), aperiodic correlation plot
% of a set of sequences X
%   X: N-by-M, a set of M unimodular sequences, each of length N
%   name: a string, if nonempty, a correlation plot is shown with the name
%   as its title
%   r: (2N-1)-by-1, the aperiodic correlation level (normalized by \sqrt{MN^2})
%   ISL: integrated sidelobe level, \sum_{k=1}^M \sum_{l=-N+1,l~=0}^{N-1}
%   |r_{kk}(l)|^2 + \sum_{k=1}^M \sum_{s=1,s!=k}^M \sum_{l=-N+1}^{N-1}
%   |r_{ks}(l)|^2
%   PSL: peak sidelobe level

[N M] = size(X);
r = zeros(2*(N-1)+1, 1);
ISL = 0;
PSL = 0;

for k = 1:(N-1)
    XShift = zeros(N, M);
    XShift((k+1):end, :) = X(1:(N-k), :);
    rMatrix = XShift' * X;
    ISL = ISL + sum((abs(rMatrix(:))).^2);
    PSL = max(PSL, max(abs(rMatrix(:))));    
    r(k+N) = norm(rMatrix, 'fro');
end
r(1:(N-1)) = r(end:-1:(N+1));
ISL = ISL * 2;

rMatrix = X' * X;
rMatrix = rMatrix - N * eye(M);
r(N) = norm(rMatrix, 'fro');
ISL = ISL + sum((abs(rMatrix(:))).^2);
PSL = max(PSL, max(abs(rMatrix(:))));

r = r / sqrt(M * N^2);

if nargin >= 2 % plot the correlation level 20*log10(r(k))
    figure;
    plot(-(N-1):(N-1), 20*log10(r));
    xlabel('Time Lag'); ylabel('Correlation Level');
    title(name);
    if nargin == 2
        V = axis;
        axis([-N+1 N-1 V(3) 0]);
    else
        axis([-N+1 N-1 y_range(1) y_range(2)]);
    end
    myboldify;
    drawnow;
    hold on;
end