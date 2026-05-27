

function X = WaveformSynthesisXoptimR(L, R, rho)
    % 输入：
    %   L   - 每个发射波形的长度
    %   R   - 期望实现的协方差矩阵，N×N
    %   rho - PAR 上界，可以是标量，也可以是 N×1 向量
    % 输出：
    %   X   - N×L 复矩阵，每一行对应一个发射波形
    %         该版本最终返回 X = sqrt(L) * R^(1/2) * U
    %         因此精确满足 X X' / L = R
    %         但 PAR 约束只近似满足
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
    Xc = zeros(N, L);
    max_iter = 1000;
    tol = 1e-8;
    obj_old = inf;
    % 交替优化
    for iter = 1:max_iter
        % Step 1：固定 U，投影得到满足 PAR 约束的 Xc
        A = sqrt(L) * Rsqrt * U;
        for n = 1:N
            c = real(R(n, n)) * L;
            x = nearestVectorAlgorithm4(A(n, :).', c, rho(n));
            Xc(n, :) = x.';
        end
        % Step 2：固定 Xc，通过 SVD 更新半酉矩阵 U
        B = sqrt(L) * Xc' * Rsqrt;
        [U1, ~, V1] = svd(B, 'econ');
        U_new = V1 * U1';
        % 收敛检查
        obj = norm(Xc - sqrt(L) * Rsqrt * U_new, 'fro')^2;
        if abs(obj_old - obj) / max(1, obj_old) < tol
            U = U_new;
            break;
        end
        U = U_new;
        obj_old = obj;
    end
    % 最终返回精确实现 R 的波形
    X = sqrt(L) * Rsqrt * U;
end



% function  X = WaveformSynthesisXoptimR(L, R, rho)
%     N = size(R, 1);
%     Rsqrt = sqrtm(R);
%     epsilon = 0.1;
%     % 交替投影
%     max_iter = 2000;
%     tol = 1e-6;
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
%         % % 收敛检查, 这里不能用这个，因为X的协方差矩阵R并不是单位阵，如果单位阵则可以用。
%         % if std(svd(X)) < tol
%         %     break;
%         % else
%         %     std(svd(X))
%         %     iter
%         %     U = U_;
%         % end
%     end
%     U = U_;
%     X = sqrt(L) * Rsqrt * U;
% end
% 
% 





