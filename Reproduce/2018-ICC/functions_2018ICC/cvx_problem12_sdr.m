


function X = cvx_problem12_sdr(H, S, X0, PT, rho)
    N = size(H, 2);
    L = size(S, 2);
    M = N * L;
    Q = rho * (H' * H) + (1 - rho) * eye(N);
    G = rho * H' * S + (1 - rho) * X0;
    A = kron(eye(L), Q);
    g = G(:);
    C = [A, -g; -g', 0];
    D = [eye(M), zeros(M, 1); zeros(1, M), 0];
    C = (C + C') / 2;
    D = (D + D') / 2;
    cvx_begin sdp quiet
        variable Z(M + 1, M + 1) hermitian semidefinite
        minimize(real(trace(C * Z)))
        subject to
            real(trace(D * Z)) == L * PT;
            Z(M + 1, M + 1) == 1;
    cvx_end
    x = Z(1:M, M + 1);
    X = reshape(x, N, L);
    X = sqrt(L * PT) * X / norm(X, 'fro');
end





