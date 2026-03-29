
function [s, k] = nearestVectorAlgorithm4(z, c, rho)
    % 算法4（来自 Tropp 等人，"通过交替投影设计结构化紧框架"）
    % 求解：min ||s - z||^2  约束：||s||^2 = c,  PAR(s) <= rho
    % 输入：
    %   z   : 长度为 d 的复列向量
    %   c   : 期望的模平方和
    %   rho : PAR 上界 (1 <= rho <= d)
    % 输出：
    %   s   : 最优向量（若问题可行）

    d = length(z);
    if rho < 1 || rho > d
        error('rho 必须满足 1 <= rho <= d');
    end
    
    % 处理全零输入
    if norm(z) == 0
        % 输入全零，任何满足能量和 PAR 的向量都是最优解，取均匀分配
        s = sqrt(c / d) * ones(d, 1);
        return;
    end
    
    delta = sqrt(c * rho / d);
    % 将 z 归一化到单位范数（算法假设如此）
    z = z / norm(z);
    a = abs(z);
    [a_sorted, idx] = sort(a);     % 升序排序
    % 预计算累积平方和以提高效率
    cum_sq = cumsum(a_sorted.^2);
    cum_sq_full = [0; cum_sq];     % cum_sq_full(i+1) = 前 i 个元素的平方和

    for k = 0:d
     
        n_rest = d - k;            % 保留的分量个数
        if n_rest == 0
            % 所有分量都被截断
            if abs(c - d*delta^2) < 1e-12
                s = delta * exp(1j * angle(z));
                return;
            else
                continue;
            end
        end
        % 唯一性检查：仅当同时有截断和保留分量时才进行
        if k > 0 && k < d && a_sorted(n_rest) == a_sorted(n_rest+1)
            continue;
        end
        rest_idx = idx(1:n_rest);   % 最小的 (d-k) 个元素的索引
        % 特殊情况：所有保留分量的幅度均为零
        if all(a(rest_idx) == 0)
            if k*delta^2 > c
                continue;
            end
            const = sqrt((c - k*delta^2) / n_rest);
            % 可行性检查：常数不能超过 delta，否则违反 PAR
            if const > delta + 1e-12
                continue;
            end
            s = zeros(d,1);
            s(rest_idx) = const * exp(1j * angle(z(rest_idx)));
            s(idx(n_rest+1:end)) = delta * exp(1j * angle(z(idx(n_rest+1:end))));
            return;
        end
        % 一般情况
        sum_rest = cum_sq_full(n_rest+1);   % 保留分量的平方和
        need = c - k*delta^2;
        if need < -1e-12
            continue;
        end
        gamma = sqrt(need / sum_rest);
        if any(gamma * a(rest_idx) > delta + 1e-12)
            continue;
        end
        % 构造解
        s = zeros(d,1);
        s(rest_idx) = gamma * z(rest_idx);
        s(idx(n_rest+1:end)) = delta * exp(1j * angle(z(idx(n_rest+1:end))));
        return;
    end

    error('算法4：未找到可行解。请检查 1 <= rho <= d 且 c > 0 是否成立。');
end