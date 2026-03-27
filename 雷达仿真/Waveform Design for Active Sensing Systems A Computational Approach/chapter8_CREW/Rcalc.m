function R = Rcalc(s, Gamma, beta, tag)
% R = Rcalc(s, Gamma, beta) or Rcalc(s, Gamma, beta, tag)
%   s: N-by-1, the probing sequence
%   Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%   beta: E{|alpha|^2} where alpha is the RCS coefficient
%   tag: 1 or 0. 0 indicates that ss' should not be included
%
%   R: N-by-N, the covariance matrix of the received sequence

if nargin < 4
    tag = 0;    % exclude ss'
end
tag = logical(tag);

N = length(s);
AH = toeplitz(s, [s(1) zeros(1,N-1) (s(N:-1:2)).']); % N-by-(2N-1)

if tag
    R = beta * (AH * AH') + Gamma;
else % this is the R we normally use
    R = beta * (AH * AH' - s * s') + Gamma;
end
