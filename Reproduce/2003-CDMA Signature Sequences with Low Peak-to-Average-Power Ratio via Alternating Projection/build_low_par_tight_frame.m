

function [X, alpha] = build_low_par_tight_frame(M, N, col_norms, rho, X0)
    % 输入：
    %   M         - 行数（波形长度）
    %   N         - 列数（用户数）
    %   col_norms - 期望列范数向量 (1×N)
    %   rho       - PAR 上界 (1 ≤ rho ≤ M)
    %   X0        - 可选初始矩阵 (M×N)，默认随机生成
    % 输出：
    %   X         - M×N 复矩阵，满足：
    %               1) X X' = α I, α = sum(col_norms.^2)/M
    %               2) ||X(:,n)|| = col_norms(n)
    %               3) PAR(X(:,n)) ≤ rho
    if nargin < 5
        X0 = [];
    end
    assert(length(col_norms) == N, 'col_norms 长度必须等于 N');
    assert(rho >= 1 && rho <= M, 'rho 必须在 [1, M] 范围内');
    
    alpha = sum(col_norms.^2) / M;   % 紧框架缩放因子
    
    % 初始化
    if isempty(X0)
        % 随机生成初始矩阵，使各列正交（近似）
        X = randn(M, N) + 1j*randn(M, N);
        [U, ~, ~] = svd(X, 'econ');
        if N <= M
            X = U(:,1:N);
        else
            % 取 U 作为前 M 列，剩余 N-M 列随机生成并正交化
            X = [U, zeros(M, N-M)];
            for n = M+1:N
                v = randn(M,1) + 1j*randn(M,1);
                v = v - X(:,1:n-1) * (X(:,1:n-1)' * v); % 正交化
                v = v / norm(v);
                X(:,n) = v;
            end
        end
    else
        X = X0;
    end
    % 调整列范数至期望值
    for n = 1:N
        X(:,n) = X(:,n) / norm(X(:,n)) * col_norms(n);
    end
    
    % 交替投影
    max_iter = 2000;
    tol = 1e-12;
    for iter = 1:max_iter
        % 1. 投影到紧框架：X X' = α I
        [U, S, V] = svd(X, 'econ');
        sigma_sum = sum(diag(S));
        X = (sigma_sum / M) * U * V';
        
        % 2. 投影到列范数和 rho 约束
        for n = 1:N
            X(:,n) = nearestVectorAlgorithm4(X(:,n), col_norms(n)^2, rho);
        end
        
        % 收敛检查
        if std(svd(X)) < tol
            break;
        end
    end
end








