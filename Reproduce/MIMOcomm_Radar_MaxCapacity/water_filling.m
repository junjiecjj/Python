function [power_allocation, water_level] = water_filling(sigma2, lamba, PT)
% 注水功率分配算法
% 输入:
%   sigma2: 噪声功率（标量）
%   lamba : 信道增益平方（特征值），向量
%   PT    : 总发射功率
% 输出:
%   power_allocation : 最优功率分配（与 lamba 顺序对应）
%   water_level      : 注水水平

    N = length(lamba);
    
    % 计算噪声归一化项
    noise_terms = sigma2 ./ lamba;
    
    % 对噪声项排序（升序），并记录原顺序
    [noise_sorted, idx_sorted] = sort(noise_terms, 'ascend');
    
    water_level = 0;
    k = 1;
    while k <= N
        % 当前可能的注水水平
        water_candidate = (PT + sum(noise_sorted(1:k))) / k;
        if k == N || water_candidate <= noise_sorted(k+1)
            water_level = water_candidate;
            break;
        end
        k = k + 1;
    end
    
    % 计算功率分配
    power_allocation = zeros(N, 1);
    for i = 1:N
        if i <= k
            power_allocation(idx_sorted(i)) = max(0, water_level - noise_sorted(i));
        else
            power_allocation(idx_sorted(i)) = 0;
        end
    end
    
    % 确保总功率精确满足约束（浮点误差微调）
    power_allocation = power_allocation * (PT / sum(power_allocation));
end