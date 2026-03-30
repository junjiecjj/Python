function z = nearestVector(z, gamma, rho)
    M = length(z);
    % 输入为行向量
    norm2 = z * z';          % 行向量点积
    if norm2 == 0
        z = sqrt(gamma) * exp(1i * angle(z));
        return;
    end
    z = sqrt(M*gamma / norm2) * z;
    beta = gamma * rho;
    if all(abs(z).^2 <= beta)
        return;
    end
    ind = true(1, M);        % 逻辑索引，标记未固定的分量
    for i = 1:M
        [~, j] = max(abs(z).^2);
        z(j) = sqrt(beta) * exp(1i * angle(z(j)));
        ind(j) = false;
        if any(ind)
            S_rem = z(ind) * z(ind)';
            if S_rem > 0
                z(ind) = sqrt((M - i*rho) * gamma / S_rem) * z(ind);
            else
                z(ind) = sqrt((M - i*rho) * gamma / sum(ind)) * exp(1i * angle(z(ind)));
            end
        end
        if all(abs(z(ind)).^2 <= beta + 1e-12)
            break;
        end
    end
end
