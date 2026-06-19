%% UKF无迹卡尔曼滤波跟踪
% 对应Python文件：25_UKF_Tracking.py
% 用途：sigma点替代Jacobian，精度优于EKF
% 运行：UKF_Tracking

function UKF_Tracking()
    %% 参数
    dt = 0.1; N = 200;
    t = (0:N-1) * dt;

    fprintf('===== UKF无迹卡尔曼滤波跟踪演示 =====\n\n');

    %% 真实轨迹：S形机动
    true_x = 500 + 10*t;
    true_y = 300 + 50*sin(0.1*t);

    %% 极坐标量测
    sigma_r = 5.0;
    sigma_theta = 2*pi/180;
    true_r = sqrt(true_x.^2 + true_y.^2);
    true_theta = atan2(true_y, true_x);
    meas_r = true_r + sigma_r * randn(1, N);
    meas_theta = true_theta + sigma_theta * randn(1, N);
    measurements_polar = [meas_r; meas_theta];

    meas_x = meas_r .* cos(meas_theta);
    meas_y = meas_r .* sin(meas_theta);

    %% UKF
    fprintf('运行UKF...\n');
    est_ukf = ukf_tracking_func(measurements_polar, dt);

    %% EKF对比
    fprintf('运行EKF...\n');
    est_ekf = ekf_tracking_func(measurements_polar, dt);

    %% RMSE
    rmse_meas = sqrt(mean((meas_x-true_x).^2 + (meas_y-true_y).^2));
    rmse_ekf = sqrt(mean((est_ekf(1,:)-true_x).^2 + (est_ekf(2,:)-true_y).^2));
    rmse_ukf = sqrt(mean((est_ukf(1,:)-true_x).^2 + (est_ukf(2,:)-true_y).^2));

    fprintf('\n原始量测RMSE: %.2f m\n', rmse_meas);
    fprintf('EKF RMSE:     %.2f m\n', rmse_ekf);
    fprintf('UKF RMSE:     %.2f m\n', rmse_ukf);

    %% 绘图
    figure('Position', [100, 100, 1200, 500]);

    subplot(1,2,1);
    plot(true_x, true_y, 'k-', 'LineWidth', 2); hold on;
    plot(meas_x, meas_y, 'g.', 'MarkerSize', 2);
    plot(est_ekf(1,:), est_ekf(2,:), 'b--', 'LineWidth', 1.5);
    plot(est_ukf(1,:), est_ukf(2,:), 'r-', 'LineWidth', 1.5);
    xlabel('X (m)'); ylabel('Y (m)');
    title('轨迹跟踪对比');
    legend('真实', '量测', sprintf('EKF (%.1fm)', rmse_ekf), ...
           sprintf('UKF (%.1fm)', rmse_ukf));
    grid on;

    subplot(1,2,2);
    err_ekf = sqrt((est_ekf(1,:)-true_x).^2 + (est_ekf(2,:)-true_y).^2);
    err_ukf = sqrt((est_ukf(1,:)-true_x).^2 + (est_ukf(2,:)-true_y).^2);
    plot(t, err_ekf, 'b--', 'LineWidth', 1.5); hold on;
    plot(t, err_ukf, 'r-', 'LineWidth', 1.5);
    xlabel('时间 (s)'); ylabel('位置误差 (m)');
    title('跟踪误差随时间变化');
    legend('EKF', 'UKF'); grid on;

    sgtitle('UKF vs EKF 跟踪性能对比', 'FontWeight', 'bold');
    saveas(gcf, 'figures/ukf_tracking.png');
    fprintf('\n演示完成\n');
end

%% ==================== UKF ====================

function est_traj = ukf_tracking_func(measurements_polar, dt)
    N = size(measurements_polar, 2);
    n = 4;

    F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
    Q = diag([0.1, 0.1, 0.5, 0.5]);
    R = diag([25, (2*pi/180)^2]);

    alpha_ = 1e-3; beta_ = 2; kappa_ = 0;
    lam = alpha_^2 * (n + kappa_) - n;

    Wm = [lam/(n+lam), repmat(1/(2*(n+lam)), 1, 2*n)];
    Wc = Wm; Wc(1) = Wm(1) + (1 - alpha_^2 + beta_);

    r0 = measurements_polar(1,1); th0 = measurements_polar(2,1);
    x = [r0*cos(th0); r0*sin(th0); 0; 0];
    P = eye(4) * 100;
    est_traj = zeros(4, N);

    for k = 1:N
        % sigma点
        sqrtP = chol((n+lam)*P, 'lower');
        X = [x, x + sqrtP, x - sqrtP];

        % 预测
        X_pred = F * X;
        x_pred = X_pred * Wm';
        P_pred = Q;
        for i = 1:(2*n+1)
            dx = X_pred(:,i) - x_pred;
            P_pred = P_pred + Wc(i) * (dx*dx');
        end

        % 量测预测
        Z_pred = zeros(2, 2*n+1);
        for i = 1:(2*n+1)
            px = X_pred(1,i); py = X_pred(2,i);
            Z_pred(:,i) = [sqrt(px^2+py^2); atan2(py,px)];
        end
        z_pred = Z_pred * Wm';

        % 协方差
        Pzz = R; Pxz = zeros(n, 2);
        for i = 1:(2*n+1)
            dz = Z_pred(:,i) - z_pred;
            dz(2) = mod(dz(2)+pi, 2*pi) - pi;
            dx = X_pred(:,i) - x_pred;
            Pzz = Pzz + Wc(i) * (dz*dz');
            Pxz = Pxz + Wc(i) * (dx*dz');
        end

        % 更新
        K = Pxz / Pzz;
        innov = measurements_polar(:,k) - z_pred;
        innov(2) = mod(innov(2)+pi, 2*pi) - pi;
        x = x_pred + K * innov;
        P = P_pred - K * Pzz * K';

        est_traj(:,k) = x;
    end
end

%% ==================== EKF（对比用） ====================

function est_traj = ekf_tracking_func(measurements_polar, dt)
    N = size(measurements_polar, 2);
    F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
    Q = diag([0.1, 0.1, 0.5, 0.5]);
    R = diag([25, (2*pi/180)^2]);

    r0 = measurements_polar(1,1); th0 = measurements_polar(2,1);
    x = [r0*cos(th0); r0*sin(th0); 0; 0];
    P = eye(4) * 100;
    est_traj = zeros(4, N);

    for k = 1:N
        x_pred = F * x;
        P_pred = F * P * F' + Q;
        px = x_pred(1); py = x_pred(2);
        r_pred = sqrt(px^2+py^2);
        z_pred = [r_pred; atan2(py,px)];
        H = [px/r_pred, py/r_pred, 0, 0;
            -py/r_pred^2, px/r_pred^2, 0, 0];
        y = measurements_polar(:,k) - z_pred;
        y(2) = mod(y(2)+pi, 2*pi) - pi;
        S = H*P_pred*H' + R;
        K = P_pred*H'/S;
        x = x_pred + K*y;
        P = (eye(4)-K*H)*P_pred;
        est_traj(:,k) = x;
    end
end
