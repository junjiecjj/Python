

function Rsqrt = hermitianSqrtPSD(R)
    R = (R + R') / 2;
    [V, D] = eig(R);
    lambda = real(diag(D));
    lambda_max = max(lambda);
    neg_tol = 1e-10 * max(1, lambda_max);
    if min(lambda) < -neg_tol
        error('R 不是半正定矩阵');
    end
    lambda(lambda < 0) = 0;
    Rsqrt = V * diag(sqrt(lambda)) * V';
    Rsqrt = (Rsqrt + Rsqrt') / 2;
end