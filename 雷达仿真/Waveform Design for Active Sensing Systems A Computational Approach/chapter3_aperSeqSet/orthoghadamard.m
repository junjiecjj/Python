function X = orthoghadamard(N, M)
% orthoghadamard: generate orthogonal waveforms scrambled with pseudo noise
%
%   X = orthoghadamard(N, M)
%   N is the number of samples, M is the number of transmitted antennas
%   X is the waveform, N-by-M
%
% 10/03/2008

if 2^(floor(log2(N))) ~= N
    error('the number of samples must be a power of 2.');
end

codeHadamard = hadamard(N); % N-by-N
XHadamard = 1/sqrt(2) * ...
    (codeHadamard(:, 1:M) - 1i * codeHadamard(:, M+1:2*M));

pseudoNoise = mseq(log2(N)); % (N-1)-by-1
X = XHadamard .* ([pseudoNoise; 1] * ones(1, M));
% X = XHadamard .* (can(N) * ones(1,M));
