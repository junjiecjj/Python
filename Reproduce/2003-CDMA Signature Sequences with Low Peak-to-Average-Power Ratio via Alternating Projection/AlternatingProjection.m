

function X  = AlternatingProjection(M, N, col_norms, rho, X0)
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
        X0 = randn(M, N) + 1j*randn(M, N);
    end
    assert(length(col_norms) == N, 'col_norms 长度必须等于 N');
    assert(rho >= 1 && rho <= M, 'rho 必须在 [1, M] 范围内');
    X = X0;
    % 调整列范数至期望值
    for n = 1:N
        X(:,n) = X(:,n) / norm(X(:,n)) * col_norms(n);
    end
    % 交替投影
    max_iter = 2000;
    tol = 1e-12;
    alpha = sum(col_norms.^2) / M;
    for iter = 1:max_iter
        % 1. 投影到紧框架：X X' = α I
        [U, S, V] = svd(X, 'econ');
        sigma_sum = sum(diag(S).^2);
        X = sqrt(sigma_sum / M) * U * V';

        % 2. 投影到列范数和 rho 约束
        for n = 1:N
            X(:,n) = nearestVectorAlgorithm4(X(:,n), col_norms(n)^2, rho);
        end

        % 收敛检查
        % err_tf = norm(X * X' - alpha * eye(M), 'fro') / norm(alpha * eye(M), 'fro');
        err_norm = max(abs(sqrt(sum(abs(X).^2, 1)) - col_norms));
        % par_vals = max(abs(X).^2, [], 1) ./ mean(abs(X).^2, 1);
        % err_par = max(par_vals - rho);
        if std(svd(X)) < tol && err_norm < tol  %  && err_par < tol   && err_tf < tol
            break;
        end
    end
end



% function X = AlternatingProjection(M, N, col_norms, rho, X0)
%     if nargin < 5 || isempty(X0)
%         X0 = randn(M, N) + 1j * randn(M, N);
%     end
%     assert(N >= M, '必须满足 N >= M，否则不可能构造 M 行满秩 tight frame');
%     assert(length(col_norms) == N, 'col_norms 长度必须等于 N');
%     assert(rho >= 1 && rho <= M, 'rho 必须在 [1, M] 范围内');
%     X = X0;
%     for n = 1:N
%         X(:, n) = X(:, n) / norm(X(:, n)) * col_norms(n);
%     end
%     max_iter = 2000;
%     tol = 1e-10;
%     alpha = sum(col_norms.^2) / M;
%     for iter = 1:max_iter
%         [U, S, V] = svd(X, 'econ');
%         sigma = diag(S);
%         sigma_bar = mean(sigma);
%         X = sigma_bar * U * V';
%         for n = 1:N
%             X(:, n) = nearestVectorAlgorithm4(X(:, n), col_norms(n)^2, rho);
%         end
%         err_tf = norm(X * X' - alpha * eye(M), 'fro') / norm(alpha * eye(M), 'fro');
%         err_norm = max(abs(sqrt(sum(abs(X).^2, 1)) - col_norms));
%         par_vals = max(abs(X).^2, [], 1) ./ mean(abs(X).^2, 1);
%         err_par = max(par_vals - rho);
%         if err_tf < tol && err_norm < tol && err_par < tol
%             break;
%         end
%     end
% end




