function [r ISL PSL] = perplotmimo(X, name)
% [r ISL PSL] = perplotmimo(X) or perplotmimo(X, name), periodic correlation plot
% of a set of sequences X
%   X: N-by-M, a set of M unimodular sequences, each of length N
%   name: a string, if nonempty, a correlation plot is shown with the name
%   as its title
%   r: (2N-1)-by-1, the periodic correlation level (normalized by \sqrt{MN^2})
%   ISL: integrated sidelobe level, \sum_{k=1}^M \sum_{l=1}^{N-1}
%   |r_{kk}(l)|^2 + \sum_{k=1}^M \sum_{s=1,s!=k}^M \sum_{l=0}^{N-1}
%   |r_{ks}(l)|^2
%   PSL: peak sidelobe level

[N M] = size(X);
r = zeros(2*(N-1)+1, 1);

rMatrix = X' * X;
rMatrix = rMatrix - N * eye(M);
r(N) = norm(rMatrix, 'fro');
ISL = sum((abs(rMatrix(:))).^2);
PSL = max(abs(rMatrix(:)));

for k = 1:(N-1)
    rMatrix = (circshift(X, k))' * X;
    ISL = ISL + sum((abs(rMatrix(:))).^2);
    PSL = max(PSL, max(abs(rMatrix(:))));
    r(k+N) = norm(rMatrix, 'fro');
end
r(1:(N-1)) = r(end:-1:(N+1));
r = r / sqrt(M * N^2);

if nargin == 2 % plot the correlation level 20*log10(r(k))
    figure;
    plot(-(N-1):(N-1), 20*log10(r));
    xlabel('lag k'); ylabel('Correlation Level');
    title(name);
    V = axis;
    axis([-N+1 N-1 V(3) 0]);
    myboldify;
    drawnow;
end