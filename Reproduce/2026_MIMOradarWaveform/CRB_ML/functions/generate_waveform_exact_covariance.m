function X = generate_waveform_exact_covariance(R, L)
    R = (R + R') / 2;
    [U, D] = eig(R);
    lambda = real(diag(D));
    idx = lambda > 1e-10;
    U = U(:, idx);
    lambda = lambda(idx);
    rankR = length(lambda);
    if L < rankR
        error('L must be no smaller than rank(R).');
    end
    G = randn(L, rankR) + 1j * randn(L, rankR);
    [Q, ~] = qr(G, 0);
    V = Q';
    F = U * diag(sqrt(lambda));
    X = sqrt(L) * F * V;
end