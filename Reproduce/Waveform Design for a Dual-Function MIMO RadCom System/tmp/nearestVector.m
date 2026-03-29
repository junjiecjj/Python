function z = nearestVector(z, gamma, rho)
% 削峰算法：将列向量 z 投影到 PAR ≤ rho 且能量 = M*gamma 的集合
    M = numel(z);
    S = z' * z;
    z = sqrt(M * gamma / S) * z;      % 缩放至能量 M*gamma
    beta = gamma * rho;                % 每个分量模平方的上限
    if all(abs(z).^2 <= beta)
        return;
    end
    ind = true(M, 1);                  % 标记未固定的分量
    for i = 1:M
        [~, j] = max(abs(z).^2);       % 找到当前幅度最大的分量
        z(j) = sqrt(beta) * exp(1i * angle(z(j))); % 截断
        ind(j) = false;
        S = z(ind)' * z(ind);          % 剩余分量当前能量
        % 重新缩放剩余分量，使其总能量 = (M - i*rho)*gamma
        z(ind) = sqrt((M - i*rho) * gamma / S) * z(ind);
        if all(abs(z(ind)).^2 <= beta + eps)
            return;
        end
    end
end