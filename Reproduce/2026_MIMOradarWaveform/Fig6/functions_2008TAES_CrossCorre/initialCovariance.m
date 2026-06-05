





function x = initialCovariance(N)
    % Create a random covariance matrix
    X = randn(N, 10*N) + 1i*randn(N, 10*N);
    L = sum(conj(X) .* X, 2);

    X = X./sqrt(L);
    R = X*X';

    M = N*(N-1);
    x = zeros(M, 1);

    % Select the elements that are above the main diagonal
    idxs = triu(ones(N, N), 1) == 1;
    x(1:M/2) = real(R(idxs));
    x(M/2+1:end) = imag(R(idxs));
end














