function X = helperCAWaveformSynthesis(R, M, rho)
    % 输出 X 为 N×M，每行一个波形
    N = size(R,1);
    epsilon = 0.1;
    P = M;
    U = (randn(P+M-1, N*P) + 1i*randn(P+M-1, N*P))/sqrt(2);
    Rexp = kron(R, eye(P));
    [L, D] = ldl(Rexp);
    Rexp_sr = sqrt(D)*L';
    maxNumIter = 1000;
    for iter = 1:maxNumIter
        Z = sqrt(M) * U * Rexp_sr;
        X_col = zeros(M, N);   % 每列一个波形，与原始代码一致
        Xexp = zeros(P+M-1, N*P);
        for n = 1:N
            zn = getzn(Z, M, P, n);
            gamma = R(n,n);
            xn = nearestVector(zn, gamma, rho);
            X_col(:, n) = xn;
            for p = 1:P
                Xexp(p:p+M-1, (n-1)*P + p) = xn;
            end
        end
        [Ubar, ~, Utilde] = svd(sqrt(M) * Rexp_sr * Xexp', 'econ');
        U_ = Utilde * Ubar';
        if norm(U_ - U) < epsilon
            break;
        else
            U = U_;
        end
    end
    X = X_col';   % 转置为 N×M，每行一个波形
end