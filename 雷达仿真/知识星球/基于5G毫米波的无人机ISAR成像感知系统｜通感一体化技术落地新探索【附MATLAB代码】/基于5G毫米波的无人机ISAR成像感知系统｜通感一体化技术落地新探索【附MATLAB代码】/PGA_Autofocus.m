%% PGA相位梯度自聚焦
% 对应Python文件：22_PGA_Autofocus.py
% 用途：不假设相位误差的参数形式，直接从数据中提取并补偿
% 运行：PGA_Autofocus

function PGA_Autofocus()
    %% 雷达参数
    fc = 28e9;
    c = 3e8;
    B = 400e6;
    Kr = B / 1e-6;
    omega = 0.5;
    PRF = 1000;
    num_pulses = 256;
    R0 = 1000;

    %% 散射点
    target_points = [0, 0, 0; 0.3, 0, 0; -0.3, 0, 0; 0, 0.25, 0];
    num_points = size(target_points, 1);

    %% 生成回波
    Tp = 1e-6;
    num_range = 256;
    fast_time = linspace(-Tp/2, Tp/2, num_range);
    slow_time = (0:num_pulses-1) / PRF;

    fprintf('===== PGA相位梯度自聚焦演示 =====\n\n');

    echo = zeros(num_range, num_pulses);
    for k = 1:num_points
        for m = 1:num_pulses
            theta = omega * slow_time(m);
            xr = target_points(k,1)*cos(theta) - target_points(k,2)*sin(theta);
            yr = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);
            Rk = R0 + yr;
            tau = 2 * yr / c;
            echo(:,m) = echo(:,m) + ...
                exp(1j*pi*Kr*(fast_time' - tau).^2) .* ...
                exp(-1j*4*pi*fc*Rk/c);
        end
    end

    %% 距离压缩
    ref = exp(-1j*pi*Kr*fast_time.^2);
    range_compressed = zeros(size(echo));
    for m = 1:num_pulses
        range_compressed(:,m) = ifft(fft(echo(:,m)) .* conj(fft(ref.')));
    end

    %% 添加相位误差
    fprintf('添加随机相位误差...\n');
    phase_err_true = 0.05*slow_time + 0.003*slow_time.^2 + ...
                     0.1*sin(2*pi*0.5*slow_time);
    data_corrupted = range_compressed .* exp(1j * phase_err_true);

    %% PGA自聚焦
    fprintf('运行PGA自聚焦 (15次迭代)...\n');
    [data_corrected, entropy_hist] = pga_autofocus_func(data_corrupted, 15);

    %% 成像
    img_ideal = fftshift(fft(range_compressed, [], 2), 2);
    img_corrupted = fftshift(fft(data_corrupted, [], 2), 2);
    img_corrected = fftshift(fft(data_corrected, [], 2), 2);

    to_db = @(x) 20*log10(abs(x) / max(abs(x(:))) + 1e-20);

    %% 绘图
    figure('Position', [100, 100, 1000, 800]);

    subplot(2,2,1);
    imagesc(to_db(img_ideal)); caxis([-40 0]); colormap jet;
    title('理想图像（无相位误差）');
    xlabel('多普勒'); ylabel('距离');

    subplot(2,2,2);
    imagesc(to_db(img_corrupted)); caxis([-40 0]); colormap jet;
    title('散焦图像（含相位误差）');
    xlabel('多普勒'); ylabel('距离');

    subplot(2,2,3);
    imagesc(to_db(img_corrected)); caxis([-40 0]); colormap jet;
    title('PGA补偿后图像');
    xlabel('多普勒'); ylabel('距离');

    subplot(2,2,4);
    plot(entropy_hist, 'b-o', 'MarkerSize', 4);
    xlabel('迭代次数'); ylabel('图像熵');
    title('PGA收敛曲线'); grid on;

    sgtitle('PGA相位梯度自聚焦效果', 'FontWeight', 'bold');

    saveas(gcf, 'figures/pga_autofocus.png');
    fprintf('\n初始图像熵: %.4f\n', entropy_hist(1));
    fprintf('最终图像熵: %.4f\n', entropy_hist(end));
    fprintf('熵降低: %.1f%%\n', (1 - entropy_hist(end)/entropy_hist(1))*100);
    fprintf('演示完成\n');
end

%% ==================== 核心算法函数 ====================

function [data_out, entropy_hist] = pga_autofocus_func(data, num_iter)
% PGA相位梯度自聚焦
% data:     num_range x num_pulses 距离压缩后数据
% num_iter: 迭代次数
% 返回: data_out 补偿后数据, entropy_hist 熵历史

    [Nr, Na] = size(data);
    data_out = data;
    entropy_hist = zeros(1, num_iter);

    for iter = 1:num_iter
        % 1. 方位FFT得到当前图像
        img = fftshift(fft(data_out, [], 2), 2);
        entropy_hist(iter) = calc_image_entropy(img);

        % 2. 选择最亮的散射点（按峰值排序，取前10%）
        peak_power = max(abs(img), [], 2);
        [~, idx] = sort(peak_power, 'descend');
        num_select = max(1, floor(Nr * 0.1));
        selected = idx(1:num_select);

        % 3. 加权平均相位梯度
        phase_gradients = zeros(1, Na-1);
        weights = zeros(1, Na-1);

        for r = selected'
            row = data_out(r, :);
            phase_diff = angle(row(2:end) .* conj(row(1:end-1)));
            w = abs(row(2:end)) .* abs(row(1:end-1));
            phase_gradients = phase_gradients + phase_diff .* w;
            weights = weights + w;
        end

        avg_gradient = phase_gradients ./ (weights + 1e-20);

        % 4. 积分得到相位误差
        phase_error = [0, cumsum(avg_gradient)];
        phase_error = phase_error - mean(phase_error);

        % 5. 全局补偿
        data_out = data_out .* exp(-1j * phase_error);
    end
end

function H = calc_image_entropy(img)
% 计算图像熵
    img_abs = abs(img);
    total = sum(img_abs(:));
    if total == 0
        H = 0; return;
    end
    p = img_abs(:) / total;
    p = p(p > 0);
    H = -sum(p .* log(p));
end
