% 测试初始化正交性
M = 3; N = 6;
rng(42);  % 固定种子

% 生成随机矩阵，取 SVD 的前 M 列作为正交基
X = randn(M, N) + 1j*randn(M, N);
[U, ~, ~] = svd(X, 'econ');
if N <= M
    X = U(:,1:N);
else
    X = [U, zeros(M, N-M)];
    for n = M+1:N
        v = randn(M,1) + 1j*randn(M,1);
        v = v - X(:,1:n-1) * (X(:,1:n-1)' * v); % 正交化
        v = v / norm(v);
        X(:,n) = v;
    end
end

% 检查正交性
fprintf('前 M 列内积（应接近 0）：\n');
for i = 1:M
    for j = i+1:M
        fprintf('(%d,%d): %e + %ei\n', i, j, real(X(:,i)'*X(:,j)), imag(X(:,i)'*X(:,j)));
    end
end

fprintf('\n后 N-M 列与前 M 列内积（应接近 0）：\n');
for i = M+1:N
    for j = 1:M
        fprintf('(%d,%d): %e + %ei\n', i, j, real(X(:,i)'*X(:,j)), imag(X(:,i)'*X(:,j)));
    end
end

fprintf('\n后 N-M 列之间内积（可以非零）：\n');
for i = M+1:N
    for j = i+1:N
        fprintf('(%d,%d): %e + %ei\n', i, j, real(X(:,i)'*X(:,j)), imag(X(:,i)'*X(:,j)));
    end
end