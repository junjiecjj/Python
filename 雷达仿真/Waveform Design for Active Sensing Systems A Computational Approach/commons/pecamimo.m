function X = pecamimo(N, M, P, X0)
% X = pecamimo(N, M, P) or pecamimo(N, M, P, X0), periodic CA MIMO
%   N: length of each transmit sequence
%   M: number of transmit sequences
%   P: only r(-P+1),...,r(0),...,r(P-1) are considered. P <= N
%   X0: N-by-M, initialization sequence set

% step 0
if nargin == 4
    X = X0;
else
    X = exp(1i * 2*pi * rand(N, M));
end

XPrev = zeros(N, M);
iterDiff = norm(X - XPrev, 'fro');
while (iterDiff > 1e-4)
    disp(iterDiff);
    XPrev = X;
    % step 2
    Z = zeros(N, M*P);
    for p = 1:P
        Z(:, p:P:(p+(M-1)*P)) = circshift(X, p-1);
    end
    [Ubar S Utilde] = svd(sqrt(N) * Z', 'econ');
    U = Utilde * Ubar';
    % step 1
    UR = U * sqrt(N); % N-by-MP
    for m = 1:M
        for n = 1:N
            rho = 0;
            ux = n; uy = (m-1)*P + 1;
            if n <= N-P+1
                for p = 0:(P-1)
                    rho = rho + UR(ux+p, uy+p);
                end
            else
                for p = 0:(N-n)
                    rho = rho + UR(ux+p, uy+p);
                end
                ux = 1; uy = uy + (N-n) + 1;
                for p = 0:(P-N+n-2)
                    rho = rho + UR(ux+p, uy+p);
                end
            end
            X(n, m) = exp(1i * angle(rho));
        end
    end
    iterDiff = norm(X - XPrev, 'fro');
end