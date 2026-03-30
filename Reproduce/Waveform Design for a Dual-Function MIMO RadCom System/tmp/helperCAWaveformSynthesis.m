function X = helperCAWaveformSynthesis(R, M, rho)
    N = size(R, 1);
    epsilon = 0.1;
    P = M;
    U = (randn(P+M-1, N*P) + 1i*randn(P+M-1, N*P)) / sqrt(2);
    Rexp = kron(R, eye(P));
    [L, D] = ldl(Rexp);
    Rexp_sr = sqrt(D) * L';
    maxNumIter = 1000;
    for iter = 1:maxNumIter
        Z = sqrt(M) * U * Rexp_sr;
        X = zeros(N, M);                % 每行一个波形
        Xexp = zeros(P+M-1, N*P);
        for n = 1:N
            zn = getzn(Z, M, P, n);     % 返回行向量
            gamma = R(n, n);
            xn = nearestVector(zn, gamma, rho); % 返回行向量
            X(n, :) = xn;
            xn_col = xn(:);              % 转为列向量用于填充 Xexp
            for p = 1:P
                Xexp(p:p+M-1, (n-1)*P + p) = xn_col;
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
end

