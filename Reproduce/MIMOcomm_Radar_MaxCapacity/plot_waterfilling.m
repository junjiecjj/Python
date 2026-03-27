function plot_waterfilling(noise_powers, optimal_powers, water_level)
% 可视化注水结果
% 输入:
%   noise_powers  : 噪声功率（基底），向量
%   optimal_powers: 分配的功率，向量
%   water_level   : 注水水平（可选，默认 1）

    if nargin < 3
        water_level = 1;
    end
    N = length(noise_powers);
    x = 1:N;
    figure;
    bar(x, noise_powers, 0.35, 'FaceColor', 'red', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    bar(x, optimal_powers, 0.35, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.7, 'BaseValue', noise_powers);
    yline(water_level, 'g--', 'LineWidth', 2);
    xlabel('信道索引');
    ylabel('功率');
    title('注水算法功率分配');
    legend('噪声功率', '分配功率', sprintf('水位线: %.3f', water_level));
    grid on;
    xticks(x);
    hold off;
end