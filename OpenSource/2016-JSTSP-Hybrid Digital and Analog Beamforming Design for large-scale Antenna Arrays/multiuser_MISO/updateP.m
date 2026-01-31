

function [P, p, lambda] = updateP(Q_tilde, beta, Pt, K, sigma2)
    % 水注法功率分配函数（使用二分法）
    % 输入参数：
    %   beta: K×1向量，用户权重系数β_k
    %   Q_tilde: K×K矩阵
    %   sigma2: 标量，噪声功率σ²
    %   P_total: 标量，总功率约束P
    % 输出参数：
    %   p: K×1向量，最优功率分配p_k（与输入顺序一致）
    %   lambda: 标量，水注水平λ

    % 提取Q_tilde的对角线元素（保持原始顺序）
    q_kk = reshape(diag(real(Q_tilde)), 1, []);
    beta = real(beta);
    % 确保beta是列向量
    % if size(beta, 2) > 1
    %     beta = beta(:);  % 转换为列向量
    % end
    % 特殊情况：总功率为0
    if Pt <= 0
        p = zeros(1, K);
        lambda = Inf;
        return;
    end
    % 确定二分法搜索区间
    % λ_min: 所有用户都有正功率时的最小值
    lambda_min = 1e-15;  % 避免除以0，使用非常小的正数
    
    % λ_max: 至少有一个用户有正功率时的最大值
    % 根据公式：β_k/λ - q_kk*σ² ≥ 0 => λ ≤ β_k/(q_kk*σ²)
    lambda_max = max(beta ./ (q_kk * sigma2));
    % 二分法参数
    max_iter = 1000;
    tolerance = 1e-10;
    % 二分法搜索
    lambda = (lambda_min + lambda_max) / 2;  % 初始化lambda为标量
    for iter = 1:max_iter
        % 计算当前λ对应的总功率
        total_power = 0;
        for k = 1:K
            term = beta(k)/lambda - q_kk(k)*sigma2;  % 这里应该是标量运算
            if term > 0
                total_power = total_power + term;  % 注意：这里计算的是q_kk * p_k
            end
        end
        % 检查收敛条件
        if abs(total_power - Pt) < tolerance
            break;
        end
        % 调整搜索区间
        if total_power < Pt
            lambda_max = lambda;  % λ太大，减少功率
        else
            lambda_min = lambda;  % λ太小，增加功率
        end
        % 更新lambda
        lambda = (lambda_min + lambda_max) / 2;
        % 防止lambda过小
        if lambda < 1e-15
            lambda = 1e-15;
            break;
        end
    end
    % 检查是否收敛
    if iter == max_iter
        warning('二分法达到最大迭代次数，可能未完全收敛');
    end
    % 计算最终功率分配
    p = max(beta/lambda - q_kk * sigma2, 0)./q_kk;
    % 验证结果
    actual_power = sum(p .* q_kk)

    % 求出P
    P = zeros(K,K);
    for kk = 1:1:K
        P(kk,kk) = p(kk);
    end
end