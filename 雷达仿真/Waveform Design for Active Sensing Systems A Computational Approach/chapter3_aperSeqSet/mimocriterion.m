function [peakAuto peakCross fitError] = mimocriterion(X, Rd, varargin)
% mimocriterion: calculate the correlation error of a MIMO waveform
%   [peakAuto peakCross fitError] = mimocriterion(X)   % Rd = N * eye(M)
%   [peakAuto peakCross fitError] = mimocriterion(X, Rd)
%   X is N-by-M, MIMO waveform
%   Rd is M-by-M, the desired covariance matrix (lag 0)
%   peakAuto and peakCross: peak sidelobe level (normalized by N)
%   fitError is the sum of the errors normalized by norm(X'*X)^2
%
%   [peakAuto peakCross fitError] = mimocriterion(X, Rd, P)
%   P-1 is the maximum lag in interest
%
%   [peakAuto peakCross fitError] = mimocriterion(X, Rd, index)
%   index is N-by-1 logical index, which indicates the lags of interest;
%
%   10/06/2008

[N M] = size(X);
if nargin == 1
    Rd = N * eye(M);
end
R = zeros(M, M, N); % R(:,:,n) is R_(n-1)
error = zeros(N, 1); % error is N-by-1, [||R0 - Rd||^2, 2||R1||^2, 2||R2||^2, ..., 2||R_(N-1)||^2]
R(:,:,1) = (X' * X).';
error(1) = (norm(R(:,:,1) - Rd, 'fro'))^2;
for n = 2:N
%     R(:,:,n) = (X' * JJ(N, n-1) * X).';
    Rtmp = zeros(M, M);
    for m1 = 1:M
        for m2 = 1:M
            x1 = zeros(N,1); x1(n:N) = X(1:(N-n+1),m1);
            x2 = X(:,m2);
            Rtmp(m1,m2) = x1' * x2;
        end
    end
    R(:,:,n) = Rtmp.';
    error(n) = 2 * (norm(R(:,:,n), 'fro'))^2;
end

R0tmp = R(:,:,1);
R0tmp(logical(diag(ones(M,1)))) = 0;
R(:,:,1) = R0tmp;
rAuto = zeros(M, N-1);
for n = 2:N
    RnTmp = R(:,:,n);
    rAuto(:,n-1) = diag(RnTmp);
    RnTmp(logical(diag(ones(M,1)))) = 0;
    R(:,:,n) = RnTmp;
end

if nargin == 3 
    if length(varargin{1}) == 1  % only P correlations considered
        P = varargin{1};
        rAuto(:,P:end) = [];
        R(:,:,(P+1):end) = [];
    else
        index = varargin{1}; % only the correlations specified by index are considered
        index = logical(index);
        rAuto(:, ~index(2:end)) = [];
        R(:,:, ~index) = [];
    end
end
peakAuto = max(abs(rAuto(:))/N);
peakCross = max(abs(R(:))/N);

if nargin <= 2
    fitError = sum(error) / (M * N^2);
elseif length(varargin{1}) == 1
    fitError = sum(error(1:P)) / (M * N^2);
else
    fitError = sum(error(index)) / (M * N^2);
end
% disp(['The fitting error: ' num2str(fitError)]);