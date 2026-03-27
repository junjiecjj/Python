function Q = Qcalc(w, Gamma, beta, tag)
% Q = Qcalc(w, Gamma, beta) or Qcalc(w, Gamma, beta, tag)
%   w: N-by-1, the receive filter
%   Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%   beta: E{|alpha|^2} where alpha is the RCS coefficient
%   tag: 1 or 0. 0 indicates that ss' should not be included
%
%   Q: N-by-N

if nargin < 4
    tag = 0;    % exclude ss'
end
tag = logical(tag);

N = length(w);
AH = toeplitz(w, [w(1) zeros(1,N-1) (w(N:-1:2)).']); % N-by-(2N-1)

if tag
    Q = beta * (AH * AH') + 1/N * w'*Gamma*w * eye(N);
else % this is the Q we normally use
    Q = beta * (AH * AH' - w * w') + 1/N * w'*Gamma*w * eye(N);
end
