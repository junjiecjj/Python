






function X = strict_waveform(H, S, Rd, L)
    N = size(H, 2);
    F = chol(Rd, 'lower');
    A = F' * H' * S;
    [U, ~, V] = svd(A, 'econ');
    VN = V(:, 1:N);
    X = sqrt(L) * F * U * VN';
end


