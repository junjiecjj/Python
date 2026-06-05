function [s, k] = nearestVectorAlgorithm4(z, c, rho)
    % Algorithm 4
    % 求解：
    %   min ||s - z||^2
    %   s.t. ||s||^2 = c, PAR(s) <= rho
    % 输入：
    %   z   : 长度为 d 的复列向量
    %   c   : 期望能量，即 ||s||^2 = c
    %   rho : PAR 上界，满足 1 <= rho <= d
    % 输出：
    %   s   : 距离 z 最近的可行向量
    %   k   : 被幅度截断到 delta 的分量个数
    d = length(z);
    eps_tol = 1e-12;
    assert(c > 0, 'c 必须为正数');
    assert(rho >= 1 && rho <= d, 'rho 必须满足 1 <= rho <= d');
    % 处理全零输入
    % 此时所有可行向量到 z 的距离都相同，取均匀分配能量的向量
    if norm(z) == 0
        s = sqrt(c / d) * ones(d, 1);
        k = 0;
        return;
    end
    % 算法4只需要 z 的方向，因此先将 z 归一化到单位范数
    z = z / norm(z);
    % 取 z 的幅度，最优 s 会保持 z 的相位，只调整幅度
    a = abs(z);
    % 固定 ||s||^2 = c 后，PAR(s) <= rho 等价于每个分量幅度不超过 delta
    delta = sqrt(c * rho / d);
    % 将幅度升序排列
    % 小幅度分量保留并整体缩放，大幅度分量截断到 delta
    [a_sorted, idx] = sort(a, 'ascend');
    % 预计算累积平方和
    % cum_sq_full(n_rest + 1) 表示前 n_rest 个最小幅度分量的平方和
    cum_sq = cumsum(a_sorted.^2);
    cum_sq_full = [0; cum_sq];
    % 枚举 k：假设最大的 k 个分量被截断到 delta
    for k = 0:d
        n_rest = d - k;
        % 特殊情况：所有分量都被截断
        if n_rest == 0
            if abs(c - d * delta^2) < eps_tol
                s = delta * exp(1j * angle(z));
                return;
            end
            continue;
        end
        % 唯一性检查
        % 如果截断边界两侧幅度相等，则该 k 对应的划分不唯一，跳过
        if k > 0 && k < d
            if abs(a_sorted(n_rest) - a_sorted(n_rest + 1)) < eps_tol
                continue;
            end
        end
        % rest_idx：未被截断的 d-k 个小幅度分量, clip_idx：被截断的 k 个大幅度分量
        rest_idx = idx(1:n_rest);
        clip_idx = idx(n_rest + 1:end);
        % 被截断的 k 个分量已经占用 k * delta^2 的能量
        % 剩余能量 need 分配给未截断分量
        need = c - k * delta^2;
        if need < -eps_tol
            continue;
        end
        % 特殊情况：保留分量的幅度全为零. 此时无法通过 gamma * z(rest_idx) 分配能量，只能均匀分配幅度
        if all(a(rest_idx) < eps_tol)
            const = sqrt(max(need, 0) / n_rest);
            if const > delta + eps_tol
                continue;
            end
            s = zeros(d, 1);
            s(rest_idx) = const * exp(1j * angle(z(rest_idx)));
            s(clip_idx) = delta * exp(1j * angle(z(clip_idx)));
            return;
        end
        % 一般情况：未截断分量保持 z 的相位和相对幅度，只乘一个公共缩放因子 gamma
        sum_rest = cum_sq_full(n_rest + 1);
        gamma = sqrt(max(need, 0) / sum_rest);
        % 检查未截断分量缩放后是否真的没有超过 delta
        if any(gamma * a(rest_idx) > delta + eps_tol)
            continue;
        end
        % 构造最终解
        % 未截断分量：gamma * z
        % 截断分量：delta * exp(j angle(z))
        s = zeros(d, 1);
        s(rest_idx) = gamma * z(rest_idx);
        s(clip_idx) = delta * exp(1j * angle(z(clip_idx)));
        return;
    end
    error('Algorithm 4 未找到可行解，请检查参数。');
end



function s = CyclicAlgorithm(z, c, rho)
    % 算法4（来自 Tropp 等人，"通过交替投影设计结构化紧框架"）
    % 求解：min ||s - z||^2  约束：||s||^2 = c,  PAR(s) <= rho
    % 输入：
    %   z   : 长度为 d 的复列向量
    %   c   : 期望的模的平方和
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














