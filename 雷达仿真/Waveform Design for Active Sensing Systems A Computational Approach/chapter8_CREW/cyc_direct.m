function [s w mse] = cyc_direct(N, Gamma, beta, rho, s0)
% [s w mse] = cyc_direct(N, Gamma, beta, rho, s0)
%   Gamma: N-by-N, the covariance matrix of interference (noise+jamming)
%   beta: E{|alpha|^2} where alpha is the RCS coefficient
%   rho: (optional) maximum allowed peak-to-average power ratio, in [1 N]
%   s0: (optional) N-by-1, sequence for initialization
%
%   s: N-by-1, the probing sequence
%   w: N-by-1, the receive filter, w = inv(R) * s
%   mse: w'Rw/(|w's|^2) = 1/(s'*inv(R)*s)

if nargin < 5
    s0 = exp(1i * 2*pi * rand(N,1));
    if nargin < 4
        rho = N;
    end
end

s = s0;
R = Rcalc(s, Gamma, beta);
spre = zeros(N, 1);
iterdiff = norm(s-spre);

while(iterdiff > 1e-2)
    spre = s;
    
    % update w
    w = R \ s; % inv(R) * s;
    
    % update s
    Q = Qcalc(w, Gamma, beta);
    s = Q \ w; % inv(Q) * w;
    s = s/norm(s) * sqrt(N);
    
    % iteration criterion
    iterdiff = norm(s - spre);
    R = Rcalc(s, Gamma, beta);
    
%     mse = 1/real(s' * (R \ s));    
%     disp(['||s-spre|| = ' num2str(iterdiff) '; MSE = ' num2str(mse)]);
end
% disp(['Before pruning, PAR = ' num2str(par(s))]);

if rho == 1
    s = exp(1i * phase(s));
elseif rho < N
    s = vectorfitpar(s, N, rho);
end

R = Rcalc(s, Gamma, beta);
w = R \ s;
mse = 1 / real(s' * w); % 1 / real(s' * inv(R) * s);
