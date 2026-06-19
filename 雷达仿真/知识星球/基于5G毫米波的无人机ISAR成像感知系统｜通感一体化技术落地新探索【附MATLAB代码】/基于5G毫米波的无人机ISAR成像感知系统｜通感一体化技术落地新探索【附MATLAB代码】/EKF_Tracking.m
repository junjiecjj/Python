%% EKF扩展卡尔曼滤波跟踪
% 对应Python文件：24_EKF_Tracking.py
% 用途：极坐标量测下的目标跟踪
% 运行：EKF_Tracking

function EKF_Tracking()
    %% 参数
    dt = 0.1;
    N = 200;
    t = (0:N-1) * dt;

    fprintf('===== EKF扩展卡尔曼滤波跟踪演示 =====\n\n');

    %% 真实轨迹：匀速圆弧
    v = 15; R_turn = 200;
    theta_traj = v * t / R_turn;
    true_x = R_turn * sin(theta_traj) + 500;
    true_y = R_turn * (1 - cos(theta_traj)) + 300;

    %% 生成极坐标量测
    sigma_r = 5.0;
    sigma_theta = 2 * pi / 180;
    true_r = sqrt(true_x.^2 + true_y.^2);
    true_theta = atan2(true_y, true_x);

    meas_r = true_r + sigma_r * randn(1, N);
    meas_theta = true_theta + sigma_theta * randn(1, N);
    measurements_polar = [meas_r; meas_theta];

    meas_x = meas_r .* cos(meas_theta);
    meas_y = meas_r .* sin(meas_theta);
    measurements_cart = [meas_x; meas_y];

    %% EKF跟踪
    fprintf('运行EKF（极坐标量测）...\n');
    est_ekf = ekf_tracking_func(measurements_polar, dt);

    %% 标准KF对比
    fprintf('运行标准KF（直角坐标量测）...\n');
    est_kf = kf_tracking_func(measurements_cart, dt);

    %% RMSE
    rmse_meas = sqrt(mean((meas_x - true_x).^2 + (meas_y - true_y).^2));
    rmse_ekf = sqrt(mean((est_ekf(1,:) - true_x).^2 + (est_ekf(2,:) - true_y).^2));
    rmse_kf = sqrt(mean((est_kf(1,:) - true_x).^2 + (est_kf(2,:) - true_y).^2));

    fprintf('\n原始量测RMSE: %.2f m\n', rmse_meas);
    fprintf('标准KF RMSE:  %.2f m\n', rmse_kf);
    fprintf('EKF RMSE:     %.2f m\n', rmse_ekf);

    %% 绘图
    figure('Position', [100, 100, 1200, 500]);

    subplot(1,2,1);
    plot(true_x, true_y, 'k-', 'LineWidth', 2); hold on;
    plot(meas_x, meas_y, 'g.', 'MarkerSize', 2);
    plot(est_kf(1,:), est_kf(2,:), 'b--', 'LineWidth', 1.5);
    plot(est_ekf(1,:), est_ekf(2,:), 'r-', 'LineWidth', 1.5);
    xlabel('X (m)'); ylabel('Y (m)');
    title('轨迹跟踪对比');
    legend('真实', '量测', sprintf('KF (%.1fm)', rmse_kf), ...
           sprintf('EKF (%.1fm)', rmse_ekf));
    grid on; axis equal;

    subplot(1,2,2);
    err_kf = sqrt((est_kf(1,:)-true_x).^2 + (est_kf(2,:)-true_y).^2);
    err_ekf = sqrt((est_ekf(1,:)-true_x).^2 + (est_ekf(2,:)-true_y).^2);
    plot(t, err_kf, 'b--', 'LineWidth', 1.5); hold on;
    plot(t, err_ekf, 'r-', 'LineWidth', 1.5);
    xlabel('时间 (s)'); ylabel('位置误差 (m)');
    title('跟踪误差随时间变化');
    legend('标准KF', 'EKF'); grid on;

    sgtitle('EKF vs 标准KF 跟踪性能对比', 'FontWeight', 'bold');
    saveas(gcf, 'figures/ekf_tracking.png');
    fprintf('\n演示完成\n');
end

%% ==================== EKF ====================

function est_traj = ekf_tracking_func(measurements_polar, dt)
    N = size(measurements_polar, 2);

    F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
    Q = diag([0.1, 0.1, 0.5, 0.5]);
    R = diag([5^2, (2*pi/180)^2]);

    r0 = measurements_polar(1,1); th0 = measurements_polar(2,1);
    x = [r0*cos(th0); r0*sin(th0); 0; 0];
    P = eye(4) * 100;
    est_traj = zeros(4, N);

    for k = 1:N
        % 预测
        x_pred = F * x;
        P_pred = F * P * F' + Q;

        % EKF更新
        px = x_pred(1); py = x_pred(2);
        r_pred = sqrt(px^2 + py^2);
        th_pred = atan2(py, px);
        z_pred = [r_pred; th_pred];

        % Jacobian
        H = [px/r_pred,  py/r_pred,  0, 0;
            -py/r_pred^2, px/r_pred^2, 0, 0];

        y = measurements_polar(:,k) - z_pred;
        y(2) = mod(y(2)+pi, 2*pi) - pi;
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x = x_pred + K * y;
        P = (eye(4) - K * H) * P_pred;
        est_traj(:, k) = x;
    end
end

%% ==================== 标准KF ====================

function est_traj = kf_tracking_func(measurements_cart, dt)
    N = size(measurements_cart, 2);

    F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
    H = [1 0 0 0; 0 1 0 0];
    Q = diag([0.1, 0.1, 0.5, 0.5]);
    R = diag([25, 25]);

    x = [measurements_cart(1,1); measurements_cart(2,1); 0; 0];
    P = eye(4) * 100;
    est_traj = zeros(4, N);

    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;

        y = measurements_cart(:,k) - H * x_pred;
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;

        x = x_pred + K * y;
        P = (eye(4) - K * H) * P_pred;
        est_traj(:, k) = x;
    end
end
