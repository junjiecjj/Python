%% Kalman滤波 - 轨迹跟踪
% Week 2使用

function [est_traj, true_traj, rmse] = module_kalman_tracking(measurements, config)
    dt = 0.1;
    N = size(measurements, 2);
    
    fprintf('  Kalman跟踪中...\n');
    fprintf('    测量点数: %d\n', N);
    
    %% 状态转移矩阵
    F = [1 0 dt 0; 
         0 1 0 dt; 
         0 0 1  0; 
         0 0 0  1];
    
    %% 测量矩阵
    H = [1 0 0 0; 
         0 1 0 0];
    
    %% 初始化
    x = [measurements(:,1); 0; 0];
    P = eye(4) * 10;
    Q = config.kalman_Q;
    R = config.kalman_R;
    
    est_traj = zeros(4, N);
    
    %% Kalman循环
    for k = 1:N
        % 预测
        x = F * x;
        P = F * P * F' + Q;
        
        % 更新
        K = P * H' / (H * P * H' + R);
        x = x + K * (measurements(:,k) - H * x);
        P = (eye(4) - K * H) * P;
        
        est_traj(:, k) = x;
    end
    
    %% 生成真实轨迹（模拟）
    t = 0:dt:(N-1)*dt;
    true_traj = [config.target_range + config.target_velocity*t;
                 5*sin(2*pi*0.5*t);
                 config.target_velocity*ones(1,N);
                 5*pi*cos(2*pi*0.5*t)];
    
    %% 计算RMSE
    errors = est_traj(1:2,:) - true_traj(1:2,:);
    rmse = sqrt(mean(sum(errors.^2, 1)));
    
    fprintf('    RMSE: %.2f 米\n', rmse);
end
