

% Eq.(19)
function a_hat = RCBfunc(at, Ryy, eps)
    % RCB: Robust Capon Beamformer
    % Input:
    %   at  - nominal steering vector (column vector, Mx1)
    %   Ryy - sample covariance matrix (M x M)
    %   eps - uncertainty sphere radius (positive scalar)
    % Output:
    %   a_hat - estimated steering vector, scaled to norm sqrt(M)

    M = length(at);
    at = at(:);                     % 确保为列向量
    I = eye(M);                     % 单位矩阵

    % 定义函数 f(lambda) = || (I + lambda*R)^{-1} at ||^2 - eps
    f = @(lam) norm((I + lam * Ryy) \ at)^2 - eps;

    % 寻找合适的 λ 区间 (f 单调递减)
    lambda_low = 0;
    f_low = f(lambda_low);
    if f_low <= 0
        % 如果 f(0) <= 0，说明 eps 太大，直接取 λ = 0
        lambda = 0;
    else
        % 增大 λ 直到 f 变为负值
        lambda_high = 1;
        while f(lambda_high) > 0 && lambda_high < 1e10
            lambda_high = lambda_high * 2;
        end
        % 二分法求解 f(lambda) = 0
        tol = 1e-8;
        max_iter = 100;
        for iter = 1:max_iter
            lambda_mid = (lambda_low + lambda_high) / 2;
            f_mid = f(lambda_mid);
            if f_mid > 0
                lambda_low = lambda_mid;
            else
                lambda_high = lambda_mid;
            end
            if abs(f_mid) < tol || (lambda_high - lambda_low) < 1e-10
                lambda = lambda_mid;
                break;
            end
        end
    end

    % 计算估计的导向矢量
    a_hat = at - (I + lambda * Ryy) \ at;

    % 缩放至范数 sqrt(M)
    a_hat = sqrt(M) * a_hat / norm(a_hat);
end