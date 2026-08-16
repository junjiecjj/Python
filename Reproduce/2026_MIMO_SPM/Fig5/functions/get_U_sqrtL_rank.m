%% Functions
function U_sqrtL = get_U_sqrtL_rank(Rs, M)
    % Rs = (1 - beta) * eye(M) + beta * ones(M);
    [U, Lambda] = eig(Rs);
    lambda = real(diag(Lambda));
    keep = lambda > 1e-10;
    U = U(:, keep);
    lambda = lambda(keep);
    U_sqrtL = U * diag(sqrt(lambda));
end