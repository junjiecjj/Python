%% 遗传算法（GA）自聚焦
% 对应Python文件：23_GA_Autofocus.py
% 用途：用遗传算法替代PSO进行最小熵ISAR图像自聚焦
% 运行：GA_Autofocus

function GA_Autofocus()
    %% 雷达参数
    fc = 28e9; c = 3e8; B = 400e6;
    Kr = B / 1e-6; omega = 0.5; PRF = 1000;
    num_pulses = 256; R0 = 1000; Tp = 1e-6;
    num_range = 256;

    target_points = [0, 0, 0; 0.3, 0, 0; -0.3, 0, 0; 0, 0.25, 0];
    num_points = size(target_points, 1);

    fast_time = linspace(-Tp/2, Tp/2, num_range);
    slow_time = (0:num_pulses-1) / PRF;

    fprintf('===== 遗传算法（GA）自聚焦演示 =====\n\n');

    %% 生成回波 + 距离压缩
    echo = zeros(num_range, num_pulses);
    for k = 1:num_points
        for m = 1:num_pulses
            theta = omega * slow_time(m);
            yr = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);
            Rk = R0 + yr;
            tau = 2 * yr / c;
            echo(:,m) = echo(:,m) + ...
                exp(1j*pi*Kr*(fast_time' - tau).^2) .* ...
                exp(-1j*4*pi*fc*Rk/c);
        end
    end

    ref = exp(-1j*pi*Kr*fast_time.^2);
    range_compressed = zeros(size(echo));
    for m = 1:num_pulses
        range_compressed(:,m) = ifft(fft(echo(:,m)) .* conj(fft(ref.')));
    end

    %% 添加相位误差
    alpha_true = 0.05; beta_true = 0.003;
    t = 0:num_pulses-1;
    phase_err = alpha_true * t + beta_true * t.^2;
    data_corrupted = range_compressed .* exp(1j * phase_err);

    %% GA自聚焦
    fprintf('真实参数: alpha=%.3f, beta=%.3f\n', alpha_true, beta_true);
    fprintf('运行GA自聚焦 (种群=30, 代数=50)...\n');
    [best_ga, hist_ga] = ga_autofocus_func(data_corrupted, 30, 50);
    fprintf('GA估计: alpha=%.6f, beta=%.6f\n', best_ga(1), best_ga(2));

    %% PSO对比
    fprintf('运行PSO自聚焦 (粒子=20, 迭代=30)...\n');
    [best_pso, hist_pso] = pso_autofocus(data_corrupted, 20, 30);
    fprintf('PSO估计: alpha=%.6f, beta=%.6f\n', best_pso(1), best_pso(2));

    %% 成像对比
    apply_corr = @(data, params) fftshift(fft( ...
        data .* exp(-1j * (params(1)*(0:size(data,2)-1) + params(2)*(0:size(data,2)-1).^2)), ...
        [], 2), 2);
    to_db = @(x) 20*log10(abs(x) / max(abs(x(:))) + 1e-20);

    img_corrupted = fftshift(fft(data_corrupted, [], 2), 2);
    img_ga = apply_corr(data_corrupted, best_ga);
    img_pso = apply_corr(data_corrupted, best_pso);

    %% 绘图
    figure('Position', [100, 100, 1000, 800]);

    subplot(2,2,1);
    imagesc(to_db(img_corrupted)); caxis([-40 0]); colormap jet;
    title('散焦图像');

    subplot(2,2,2);
    imagesc(to_db(img_ga)); caxis([-40 0]); colormap jet;
    title(sprintf('GA补偿 (\\alpha=%.4f, \\beta=%.5f)', best_ga(1), best_ga(2)));

    subplot(2,2,3);
    imagesc(to_db(img_pso)); caxis([-40 0]); colormap jet;
    title(sprintf('PSO补偿 (\\alpha=%.4f, \\beta=%.5f)', best_pso(1), best_pso(2)));

    subplot(2,2,4);
    plot(hist_ga, 'r-', 'LineWidth', 1.5); hold on;
    plot(hist_pso, 'b-', 'LineWidth', 1.5);
    xlabel('迭代/代数'); ylabel('最优图像熵');
    title('GA vs PSO 收敛曲线');
    legend('GA', 'PSO'); grid on;

    sgtitle('GA vs PSO 自聚焦对比', 'FontWeight', 'bold');
    saveas(gcf, 'figures/ga_vs_pso.png');
    fprintf('\n演示完成\n');
end

%% ==================== GA核心 ====================

function [best_params, history] = ga_autofocus_func(data, pop_size, max_gen)
    dim = 2;
    lb = [-0.1, -0.01]; ub = [0.1, 0.01];

    pop = lb + (ub - lb) .* rand(pop_size, dim);
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = calc_entropy(pop(i,:), data);
    end

    [best_val, best_idx] = min(fitness);
    best_params = pop(best_idx, :);
    history = zeros(max_gen, 1);

    for gen = 1:max_gen
        % 锦标赛选择（保留精英）
        new_pop = zeros(size(pop));
        new_pop(1,:) = best_params;
        for i = 2:pop_size
            a = randi(pop_size); b = randi(pop_size);
            if fitness(a) < fitness(b)
                new_pop(i,:) = pop(a,:);
            else
                new_pop(i,:) = pop(b,:);
            end
        end

        % 算术交叉
        for i = 2:2:pop_size-1
            if rand() < 0.8
                alpha_c = rand();
                c1 = alpha_c*new_pop(i,:) + (1-alpha_c)*new_pop(i+1,:);
                c2 = (1-alpha_c)*new_pop(i,:) + alpha_c*new_pop(i+1,:);
                new_pop(i,:) = c1;
                new_pop(i+1,:) = c2;
            end
        end

        % 高斯变异
        for i = 2:pop_size
            if rand() < 0.1
                new_pop(i,:) = new_pop(i,:) + randn(1,dim)*0.01;
                new_pop(i,:) = max(lb, min(ub, new_pop(i,:)));
            end
        end

        pop = new_pop;
        for i = 1:pop_size
            fitness(i) = calc_entropy(pop(i,:), data);
        end

        [gen_best, gen_idx] = min(fitness);
        if gen_best < best_val
            best_val = gen_best;
            best_params = pop(gen_idx, :);
        end
        history(gen) = best_val;
    end
end

%% ==================== PSO（对比用） ====================

function [gbest, history] = pso_autofocus(data, np_, ni)
    lb = [-0.1, -0.01]; ub = [0.1, 0.01];
    dim = 2; w = 0.7; c1 = 1.5; c2 = 1.5;

    pos = lb + (ub - lb) .* rand(np_, dim);
    vel = zeros(np_, dim);
    fit = zeros(np_, 1);
    for i = 1:np_
        fit(i) = calc_entropy(pos(i,:), data);
    end

    pbest = pos; pbest_fit = fit;
    [gval, gidx] = min(fit);
    gbest = pos(gidx,:); gbest_fit = gval;
    history = zeros(ni, 1);

    for iter = 1:ni
        r1 = rand(np_, dim); r2 = rand(np_, dim);
        vel = w*vel + c1*r1.*(pbest-pos) + c2*r2.*(gbest-pos);
        pos = max(lb, min(ub, pos + vel));

        for i = 1:np_
            fit(i) = calc_entropy(pos(i,:), data);
        end

        better = fit < pbest_fit;
        pbest(better,:) = pos(better,:);
        pbest_fit(better) = fit(better);

        [mv, mi] = min(fit);
        if mv < gbest_fit
            gbest_fit = mv; gbest = pos(mi,:);
        end
        history(iter) = gbest_fit;
    end
end

%% ==================== 图像熵 ====================

function H = calc_entropy(params, data)
    t = 0:size(data,2)-1;
    phase = params(1)*t + params(2)*t.^2;
    corrected = data .* exp(-1j * phase);
    img = fftshift(fft(corrected, [], 2), 2);

    img_abs = abs(img) / sum(abs(img(:)));
    img_abs(img_abs == 0) = eps;
    H = -sum(img_abs(:) .* log(img_abs(:)));
end
