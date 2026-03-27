function X = camimo(N, M, P, X0)
% X = camimo(N, M, P) or X = camimo(N, M, P, X0), the CA algorithm
%   N: length of each transmit sequence
%   M: number of transmit sequences
%   P: only r(-P+1),...,r(0),...,r(P-1) are considered. P <= N
%   X0: N-by-M, initialization sequence set
%   X: N-by-M, the designed sequence set

if nargin == 4
    X = X0;
else
    X = exp(1i * 2*pi * rand(N,M));
end

% % if considering a non-identity covariance matrix, replace Rd with the
% % desired covariance matrix and replace sqrt(N) with RdTildeRoot in <--
% Rd = N * eye(M);
% RdTilde = kron(Rd, eye(P)); % MP-by-MP
% RdTildeRoot = sqrtm(RdTilde);

% step 0
XPrev = zeros(N,M);
iterDiff = norm(X - XPrev, 'fro');

while (iterDiff > 1e-3)
    disp(iterDiff);
    XPrev = X;
    % step 2
    Z = zeros(N+P-1, M*P);
    for p = 1:P
        Z(p:(p+N-1), p:P:(p+(M-1)*P)) = X;
    end
    [Ubar S Utilde] = svd(sqrt(N) * Z', 'econ'); % <--
    U = Utilde * Ubar'; % (N+P-1)-by-(MP)
    % step 1
    UR = U * sqrt(N); % <--
    for m = 1:M
        for n = 1:N
            rho = 0;
            ux = n; uy = (m-1)*P + 1;
            for p = 0:(P-1)
                rho = rho + UR(ux+p, uy+p);
            end
            X(n, m) = exp(1i * angle(rho));
        end
    end
    iterDiff = norm(X - XPrev, 'fro');
end