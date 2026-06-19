%% PSO优化算法 - 图像熵最小化
% 用于优化ISAR图像质量
% Week 2使用

function [img_opt, best_params, history] = module_pso_optimization(data, config)
    %% PSO参数
    num_particles = config.pso_particles;
    num_iterations = config.pso_iterations;

    % 搜索空间
    dim = 2;
    lb = [-0.1, -0.01];
    ub = [0.1, 0.01];

    fprintf('  PSO优化中...\n');
    fprintf('    粒子数: %d\n', num_particles);
    fprintf('    迭代数: %d\n', num_iterations);

    %% 定义目标函数
    objective = @(x) calc_entropy(x, data);

    %% 使用手动PSO实现（保证记录完整迭代历史）
    [best_params, best_fitness, history] = pso_manual(objective, dim, lb, ub, ...
                                                      num_particles, num_iterations);

    %% 应用最优参数
    t = 1:size(data,2);
    phase_corr = best_params(1)*t + best_params(2)*t.^2;
    corrected = data .* exp(-1j * phase_corr);
    img_opt = fftshift(fft(corrected, [], 2), 2);

    fprintf('    最优熵值: %.4f\n', best_fitness);
end

function entropy = calc_entropy(params, data)
    % 计算图像熵
    t = 1:size(data,2);
    phase = params(1)*t + params(2)*t.^2;
    corrected = data .* exp(-1j * phase);
    img = fftshift(fft(corrected, [], 2), 2);

    img_abs = abs(img) / sum(abs(img(:)));
    img_abs(img_abs == 0) = eps;
    entropy = -sum(img_abs(:) .* log(img_abs(:)));
end

function [best_pos, best_val, history] = pso_manual(objective, dim, lb, ub, ...
                                                     num_particles, num_iterations)
    % 手动实现PSO（不依赖工具箱）
    w = 0.7;       % 惯性权重
    c1 = 1.5;      % 个体学习因子
    c2 = 1.5;      % 社会学习因子

    % 初始化粒子
    positions = lb + (ub - lb) .* rand(num_particles, dim);
    velocities = zeros(num_particles, dim);

    % 评估初始适应度
    fitness = zeros(num_particles, 1);
    for i = 1:num_particles
        fitness(i) = objective(positions(i, :));
    end

    % 初始化个体最优和全局最优
    pbest_pos = positions;
    pbest_val = fitness;
    [best_val, best_idx] = min(fitness);
    best_pos = positions(best_idx, :);

    history = zeros(num_iterations, 1);

    for iter = 1:num_iterations
        for i = 1:num_particles
            % 更新速度
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            velocities(i,:) = w * velocities(i,:) + ...
                              c1 * r1 .* (pbest_pos(i,:) - positions(i,:)) + ...
                              c2 * r2 .* (best_pos - positions(i,:));

            % 更新位置
            positions(i,:) = positions(i,:) + velocities(i,:);

            % 边界约束
            positions(i,:) = max(lb, min(ub, positions(i,:)));

            % 评估适应度
            fitness(i) = objective(positions(i,:));

            % 更新个体最优
            if fitness(i) < pbest_val(i)
                pbest_val(i) = fitness(i);
                pbest_pos(i,:) = positions(i,:);
            end

            % 更新全局最优
            if fitness(i) < best_val
                best_val = fitness(i);
                best_pos = positions(i,:);
            end
        end

        history(iter) = best_val;
    end
end
