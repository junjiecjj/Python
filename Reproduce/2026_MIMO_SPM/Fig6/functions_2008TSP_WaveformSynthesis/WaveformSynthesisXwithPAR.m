function X = WaveformSynthesisXwithPAR(R, L, rho)
    % 输入：
    %   L   - 每个发射波形的长度
    %   R   - 期望逼近的协方差矩阵，N×N
    %   rho - PAR 上界，可以是标量，也可以是 N×1 向量
    % 输出：
    %   X   - N×L 复矩阵，每一行对应一个发射波形
    %         该版本最终返回 PAR 投影后的 X
    %         因此严格满足每行的能量和 PAR 约束
    %         但一般只近似满足 X X' / L ≈ R
    N = size(R, 1);
    assert(L >= N, '必须满足 L >= N');
    if isscalar(rho)
        rho = rho * ones(N, 1);
    else
        rho = rho(:);
    end
    R = (R + R') / 2;
    Rsqrt = hermitianSqrtPSD(R);
    % 初始化半酉矩阵 U，使 U U' = I
    U = initRowSemiUnitary(N, L);
    X = zeros(N, L);
    max_iter = 1000;
    tol = 1e-8;
    obj_old = inf;
    % 交替优化
    for iter = 1:max_iter
        % Step 1：固定 U，逐行投影到能量约束和 PAR 约束集合
        A = sqrt(L) * Rsqrt * U;
        for n = 1:N
            c = real(R(n, n)) * L;
            x = nearestVectorAlgorithm4(A(n, :).', c, rho(n));
            X(n, :) = x.';
        end
        % Step 2：固定 X，通过 SVD 更新半酉矩阵 U
        B = sqrt(L) * X' * Rsqrt;
        [U1, ~, V1] = svd(B, 'econ');
        U_new = V1 * U1';
        % 收敛检查
        obj = norm(X - sqrt(L) * Rsqrt * U_new, 'fro')^2;
        if abs(obj_old - obj) / max(1, obj_old) < tol
            U = U_new;
            break;
        end
        U = U_new;
        obj_old = obj;
    end
end


% function X = WaveformSynthesisXwithPAR(L, R, rho  )
% 
%     N = size(R, 1);
%     Rsqrt = sqrtm(R);
%     epsilon = 0.1;
%     % 交替投影
%     max_iter = 2000;
%     tol = 1e-12;
%     U = randn(N, L) + 1i * randn(N, L);
% 
%     X = zeros(N, L);
%     for iter = 1:max_iter
%         % Step 1, 固定U, 更新X
%         A = sqrt(L) * Rsqrt * U;
%         for n = 1:N
%             xn = nearestVectorAlgorithm4(A(n,:).', R(n,n)*L, rho);
%             X(n, :) = xn.';
%         end
% 
%         % Step 2. 固定X, 更新U
%         [U1, ~, V1] = svd(sqrt(L) * X' * Rsqrt, 'econ');
%         U_ = V1 * U1';
%         % 收敛检查
%         if norm(U_ - U) < epsilon
%             break;
%         else
%             U = U_;
%         end
% 
%         % % 收敛检查
%         % if std(svd(X)) < tol
%         %     break;
%         % else
%         %     U = U_;
%         % end
% 
%     end
% end







