





function F = constrainedCovariance(x, N)
    % Reconstruct the covariance matrix from a vector x of above diagonal
    % values. The diagonal elements are all equal to 1.
    Re = zeros(N, N);
    Im = zeros(N, N);
    M = numel(x);

    idxs = triu(ones(N, N), 1) == 1;
    Re(idxs) = x(1:M/2);
    Im(idxs) = x(M/2+1:end);

    F = eye(N, N);
    F = F + Re + Re.' + 1i*Im - 1i*Im.';
end
















