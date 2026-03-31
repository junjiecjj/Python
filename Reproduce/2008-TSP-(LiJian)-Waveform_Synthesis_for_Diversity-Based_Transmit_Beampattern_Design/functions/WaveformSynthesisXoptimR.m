









function  X = WaveformSynthesisXoptimR(L, R, rho)
    N = size(R, 1);
    Rsqrt = sqrtm(R);
    epsilon = 0.1;
    % 交替投影
    max_iter = 2000;
    tol = 1e-6;
    U = randn(N, L) + 1i * randn(N, L);
    
    X = zeros(N, L);
    for iter = 1:max_iter
        % Step 1, 固定U, 更新X
        A = sqrt(L) * Rsqrt * U;
        for n = 1:N
            xn = CyclicAlgorithm(A(n,:).', R(n,n)*L, rho);
            X(n, :) = xn.';
        end

        % Step 2. 固定X, 更新U
        [U1, ~, V1] = svd(sqrt(L) * X' * Rsqrt, 'econ');
        U_ = V1 * U1';
        % 收敛检查
        if norm(U_ - U) < epsilon
            break;
        else
            U = U_;
        end

        % % 收敛检查, 这里不能用这个，因为X的协方差矩阵R并不是单位阵，如果单位阵则可以用。
        % if std(svd(X)) < tol
        %     break;
        % else
        %     std(svd(X))
        %     iter
        %     U = U_;
        % end
        
    end
    U = U_;
    X = sqrt(L) * Rsqrt * U;
end







