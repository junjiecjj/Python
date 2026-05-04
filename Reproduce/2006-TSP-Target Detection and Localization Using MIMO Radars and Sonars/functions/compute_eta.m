%% 辅助函数：计算等效观测 eta（基于充分统计量）
% 输入：y (Mx1), s (Mx1), R_s, N=1
% 输出：eta (M^2 x 1)
function eta = compute_eta(y, s, R_s, N)
    E = (1/sqrt(N)) * (y * s');   % MxM
    [U, Lambda] = eig(R_s);
    % 方法 A：使用 pinv（适合通用矩阵）
    U_tmp = U * pinv(sqrtm(Lambda));
    
    % 方法 B：显式处理零特征值（效率更高）
    % tol = 1e-12;
    % lambda_sqrt = sqrt(max(lambda, 0));
    % inv_lambda_sqrt = zeros(size(lambda));
    % inv_lambda_sqrt(lambda > tol) = 1 ./ lambda_sqrt(lambda > tol);
    % U_tmp = U * diag(inv_lambda_sqrt);
    eta = reshape(E * U_tmp, [], 1);
end