function X = wecanmimo(N, M, gamma, X0)
% X = wecanmimo(N, M, gamma) or X = wecanmimo(N, M, gamma, X0), WeCAN MIMO
%   N: length of each transmit sequence
%   M: number of transmit sequences
%   gamma: N-by-1, corresponding to weights w_k = gamma_k^2
%   X0: N-by-M, initialization sequence set

if nargin == 4
    X = X0;
else
    X = exp(1i * 2*pi * rand(N, M));
end
XPrev = zeros(N, M);
iterDiff = norm(X - XPrev, 'fro');

gamma(1) = 0;
Gamma = toeplitz(gamma);
eigvalues = eig(Gamma);
gamma(1) = abs(min(eigvalues)) + 0.01;
Gamma = toeplitz(gamma)/gamma(1);

C = sqrtm(Gamma);

Xtilde = zeros(N, N*M);
U = zeros(N, M, 2*N);
G = zeros(2*N, N*M);

while (iterDiff > 1e-3)
    disp(iterDiff);
    XPrev = X;
    % step 1, X is given
    for m = 1:M
        Xtilde(:, (m-1)*N+1:m*N) = C .* repmat(X(:,m), [1 N]);
    end
    F = fft([Xtilde; zeros(N, N*M)]); % 2N-by-NM
    for p = 1:(2*N)
        [U1 S U2] = svd(sqrt(N) * (reshape(F(p,:), [N M]))', 'econ');
        U(:,:,p) = U2 * U1'; % N-by-M
    end
    % step 2, U is given
    for p = 1:(2*N)
        G(p,:) = reshape(U(:,:,p)*sqrt(N)*sqrt(Gamma(1,1)), [1 N*M]);
    end
    V = ifft(G); % 2N-by-NM
    W = repmat(C, [1 M]) .* V(1:N, :); % N-by-NM
    for m = 1:M
        mx = (m-1)*N+1; my = m*N;
        for n = 1:N
            phi = angle(sum(W(n, mx:my)));
            X(n, m) = exp(1i * phi);
        end
    end
    iterDiff = norm(X - XPrev, 'fro');
end