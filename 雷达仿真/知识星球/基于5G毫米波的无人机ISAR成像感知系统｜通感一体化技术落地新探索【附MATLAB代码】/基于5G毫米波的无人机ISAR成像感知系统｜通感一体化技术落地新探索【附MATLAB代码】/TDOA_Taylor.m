%% Taylor级数TDOA迭代定位
% 对应Python文件：26_TDOA_Taylor.py
% 用途：Taylor展开迭代精化TDOA定位，精度高于Chan算法
% 运行：TDOA_Taylor

function TDOA_Taylor()
    c = 3e8;

    %% 基站布局（四面体）
    stations = [0, 0, 0; 1000, 0, 0; 500, 866, 0; 500, 289, 816];

    %% 真实目标位置
    true_pos = [400; 350; 200];

    fprintf('===== Taylor级数TDOA迭代定位演示 =====\n\n');
    fprintf('真实位置: [%.0f, %.0f, %.0f]\n', true_pos);
    fprintf('基站数量: %d\n', size(stations, 1));

    %% 蒙特卡洛仿真
    num_trials = 200;
    noise_levels = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8];

    errors_chan = zeros(1, length(noise_levels));
    errors_taylor = zeros(1, length(noise_levels));

    fprintf('\n运行蒙特卡洛仿真 (%d 次/噪声级)...\n', num_trials);

    for ni = 1:length(noise_levels)
        sigma_t = noise_levels(ni);
        chan_err = zeros(num_trials, 1);
        taylor_err = zeros(num_trials, 1);

        for trial = 1:num_trials
            % 生成带噪声时差
            true_dists = zeros(size(stations, 1), 1);
            for i = 1:size(stations, 1)
                true_dists(i) = norm(true_pos - stations(i,:)');
            end
            true_tdiffs = (true_dists(2:end) - true_dists(1)) / c;
            noisy_tdiffs = true_tdiffs + sigma_t * randn(size(true_tdiffs));

            % Chan算法
            pos_chan = tdoa_chan_func(stations, noisy_tdiffs, c);
            chan_err(trial) = norm(pos_chan - true_pos);

            % Taylor迭代
            pos_taylor = tdoa_taylor_func(stations, noisy_tdiffs, c, 20);
            taylor_err(trial) = norm(pos_taylor - true_pos);
        end

        errors_chan(ni) = median(chan_err);
        % 过滤发散离群值后取中位数
        valid = taylor_err < 1e6;
        if any(valid)
            errors_taylor(ni) = median(taylor_err(valid));
        else
            errors_taylor(ni) = NaN;
        end

        range_err = sigma_t * c;
        fprintf('  噪声=%.1fns (~%.1fm): Chan=%.2fm, Taylor=%.2fm\n', ...
            sigma_t*1e9, range_err, errors_chan(ni), errors_taylor(ni));
    end

    %% 绘图
    figure('Position', [100, 100, 1200, 500]);

    range_errors = noise_levels * c;

    subplot(1,2,1);
    semilogy(range_errors, errors_chan, 'b-o', 'LineWidth', 1.5); hold on;
    semilogy(range_errors, errors_taylor, 'r-s', 'LineWidth', 1.5);
    xlabel('等效距离噪声 (m)'); ylabel('平均定位误差 (m)');
    title('定位精度 vs 噪声水平');
    legend('Chan算法', 'Taylor迭代'); grid on;

    % 单次可视化
    subplot(1,2,2);
    sigma_t = 5e-10;
    true_dists = zeros(size(stations,1), 1);
    for i = 1:size(stations,1)
        true_dists(i) = norm(true_pos - stations(i,:)');
    end
    true_tdiffs = (true_dists(2:end) - true_dists(1)) / c;
    noisy_tdiffs = true_tdiffs + sigma_t * randn(size(true_tdiffs));

    pos_c = tdoa_chan_func(stations, noisy_tdiffs, c);
    pos_t = tdoa_taylor_func(stations, noisy_tdiffs, c, 20);

    scatter(stations(:,1), stations(:,2), 100, 'k', '^', 'filled'); hold on;
    plot(true_pos(1), true_pos(2), 'gp', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
    plot(pos_c(1), pos_c(2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(pos_t(1), pos_t(2), 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    xlabel('X (m)'); ylabel('Y (m)');
    title('单次定位结果（俯视图）');
    legend('基站', '真实', 'Chan', 'Taylor'); grid on;

    sgtitle('Taylor级数TDOA vs Chan算法', 'FontWeight', 'bold');
    saveas(gcf, 'figures/tdoa_taylor.png');
    fprintf('\n演示完成\n');
end

%% ==================== Taylor TDOA ====================

function pos = tdoa_taylor_func(stations, time_diffs, c, max_iter)
    N = size(stations, 1);
    s1 = stations(1,:)';

    % 初始猜测：基站几何中心
    pos = mean(stations, 1)';

    % Taylor迭代精化（含发散检测）
    prev_norm = inf;
    for iter = 1:max_iter
        d = zeros(N, 1);
        for i = 1:N
            d(i) = norm(pos - stations(i,:)');
        end
        if any(d < 1e-6), break; end

        H = zeros(N-1, 3);
        delta_r = zeros(N-1, 1);
        for i = 2:N
            si = stations(i,:)';
            H(i-1,:) = ((pos-si)/d(i) - (pos-s1)/d(1))';
            delta_r(i-1) = c*time_diffs(i-1) - (d(i) - d(1));
        end

        delta_pos = (H'*H) \ (H'*delta_r);
        curr_norm = norm(delta_pos);
        if curr_norm > prev_norm * 10, break; end  % 发散检测
        prev_norm = curr_norm;
        pos = pos + delta_pos;

        if curr_norm < 1e-4, break; end
    end
end

%% ==================== Chan TDOA ====================

function pos = tdoa_chan_func(stations, time_diffs, c)
    N = size(stations, 1);
    s1 = stations(1,:)';

    A = zeros(N-1, 3); b = zeros(N-1, 1);
    for i = 2:N
        si = stations(i,:)';
        A(i-1,:) = 2*(si - s1)';
        b(i-1) = c^2*time_diffs(i-1)^2 - si'*si + s1'*s1;
    end
    pos = (A'*A) \ (A'*b);
end
