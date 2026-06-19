%% 多基站协同定位系统
% =============================================
% 研究方向：定位导航技术
% 
% 功能：
% 1. TDOA/AOA/RSS定位算法
% 2. 多基站信息融合
% 3. 卡尔曼滤波/粒子滤波跟踪
% 4. GDOP分析
% =============================================

classdef MultiStationLocalization < handle
    
    properties
        base_stations    % 基站位置 [x, y, z]
        measurements     % 测量数据
        target_state     % 目标状态估计
        localization_method  % 定位方法
        metrics          % 性能指标
    end
    
    methods
        function obj = MultiStationLocalization(stations)
            % 构造函数
            % stations: Nx3矩阵，每行是基站位置[x, y, z]
            obj.base_stations = stations;
            obj.metrics = struct();
        end
        
        function pos = localizeTDOA(obj, time_differences, c)
            % TDOA定位（到达时间差）
            % time_differences: (N-1)x1向量，相对于第一个基站的时差
            % c: 光速
            
            fprintf('TDOA定位中...\n');
            
            N = size(obj.base_stations, 1);
            
            % 构建方程 Ax = b
            % (d_i - d_1)^2 = c^2 * tau_i^2
            A = zeros(N-1, 3);
            b = zeros(N-1, 1);
            
            s1 = obj.base_stations(1, :);  % 参考基站
            
            for i = 2:N
                si = obj.base_stations(i, :);
                tau_i = time_differences(i-1);
                
                A(i-1, :) = 2 * (si - s1);
                b(i-1) = c^2 * tau_i^2 - norm(si)^2 + norm(s1)^2;
            end
            
            % 最小二乘求解
            pos = (A' * A) \ (A' * b);
            
            % 精化（非线性优化）
            objective = @(p) obj.tdoaObjective(p, time_differences, c);
            try
                options = optimoptions('fminunc', 'Display', 'off');
                pos = fminunc(objective, pos, options);
            catch
                opts = optimset('MaxIter', 500, 'Display', 'off');
                pos = fminsearch(objective, pos, opts);
            end
            
            fprintf('  估计位置: [%.2f, %.2f, %.2f]\n', pos);
        end
        
        function cost = tdoaObjective(obj, pos, tau, c)
            % TDOA目标函数
            N = size(obj.base_stations, 1);
            cost = 0;
            
            d1 = norm(pos - obj.base_stations(1, :));
            
            for i = 2:N
                di = norm(pos - obj.base_stations(i, :));
                measured_diff = c * tau(i-1);
                predicted_diff = di - d1;
                cost = cost + (measured_diff - predicted_diff)^2;
            end
        end
        
        function pos = localizeAOA(obj, angles)
            % AOA定位（到达角）— 解析解
            % angles: Nx2矩阵 [azimuth, elevation]（度）

            fprintf('AOA定位中...\n');

            N = size(obj.base_stations, 1);

            % 转换为弧度
            az = deg2rad(angles(:, 1));
            el = deg2rad(angles(:, 2));

            % 构建方向向量
            directions = zeros(N, 3);
            for i = 1:N
                directions(i, :) = [
                    cos(el(i)) * cos(az(i));
                    cos(el(i)) * sin(az(i));
                    sin(el(i))
                ];
            end

            % 解析解：p = A \ b
            % 其中 M_i = I - d_i * d_i' （垂直投影矩阵）
            %      A = sum(M_i),  b = sum(M_i * s_i)
            I3 = eye(3);
            A = zeros(3);
            b = zeros(3, 1);

            for i = 1:N
                di = directions(i, :)';
                si = obj.base_stations(i, :)';
                Mi = I3 - di * di';   % 投影矩阵（对称幂等）
                A = A + Mi;
                b = b + Mi * si;
            end

            pos = (A \ b)';

            fprintf('  估计位置: [%.2f, %.2f, %.2f]\n', pos);
        end
        
        function pos = localizeRSS(obj, rss_values, path_loss_params)
            % RSS定位（接收信号强度）
            % rss_values: Nx1向量（dBm）
            % path_loss_params: struct with P0, n
            
            fprintf('RSS定位中...\n');
            
            P0 = path_loss_params.P0;  % 参考功率
            n = path_loss_params.n;     % 路径损耗指数
            
            % RSS模型: RSS_i = P0 - 10*n*log10(d_i/d0)
            % 转换为距离
            d0 = 1;  % 参考距离
            distances = d0 * 10.^((P0 - rss_values) / (10*n));
            
            % 三边定位
            pos = obj.trilateration(distances);
            
            fprintf('  估计位置: [%.2f, %.2f, %.2f]\n', pos);
        end
        
        function pos = trilateration(obj, distances)
            % 三边定位
            N = size(obj.base_stations, 1);
            
            % 构建方程
            A = zeros(N-1, 3);
            b = zeros(N-1, 1);
            
            s1 = obj.base_stations(1, :);
            d1 = distances(1);
            
            for i = 2:N
                si = obj.base_stations(i, :);
                di = distances(i);
                
                A(i-1, :) = 2 * (si - s1);
                b(i-1) = di^2 - d1^2 - norm(si)^2 + norm(s1)^2;
            end
            
            % 最小二乘
            pos = (A' * A) \ (A' * b);
            
            % 精化
            objective = @(p) obj.trilaterationObjective(p, distances);
            try
                options = optimoptions('fminunc', 'Display', 'off');
                pos = fminunc(objective, pos, options);
            catch
                opts = optimset('MaxIter', 500, 'Display', 'off');
                pos = fminsearch(objective, pos, opts);
            end
        end
        
        function cost = trilaterationObjective(obj, pos, distances)
            % 三边定位目标函数
            N = size(obj.base_stations, 1);
            cost = 0;
            
            for i = 1:N
                predicted_dist = norm(pos - obj.base_stations(i, :));
                cost = cost + (predicted_dist - distances(i))^2;
            end
        end
        
        function pos = hybridLocalization(obj, tdoa_data, aoa_data, rss_data)
            % 混合定位（融合多种测量）
            fprintf('混合定位中...\n');
            
            % 分别计算
            if ~isempty(tdoa_data)
                pos_tdoa = obj.localizeTDOA(tdoa_data.time_diff, tdoa_data.c);
                weight_tdoa = 1.0;
            else
                pos_tdoa = zeros(3, 1);
                weight_tdoa = 0;
            end
            
            if ~isempty(aoa_data)
                pos_aoa = obj.localizeAOA(aoa_data.angles);
                weight_aoa = 0.5;
            else
                pos_aoa = zeros(3, 1);
                weight_aoa = 0;
            end
            
            if ~isempty(rss_data)
                pos_rss = obj.localizeRSS(rss_data.rss, rss_data.params);
                weight_rss = 0.3;
            else
                pos_rss = zeros(3, 1);
                weight_rss = 0;
            end
            
            % 加权融合
            total_weight = weight_tdoa + weight_aoa + weight_rss;
            pos = (weight_tdoa * pos_tdoa + weight_aoa * pos_aoa + weight_rss * pos_rss) / total_weight;
            
            fprintf('  融合位置: [%.2f, %.2f, %.2f]\n', pos);
        end
        
        function gdop = calculateGDOP(obj, target_pos)
            % 计算GDOP（几何精度衰减因子）
            N = size(obj.base_stations, 1);
            
            % 几何矩阵
            G = zeros(N, 4);
            for i = 1:N
                si = obj.base_stations(i, :);
                r = norm(target_pos - si);
                G(i, :) = [-(target_pos - si) / r, 1];
            end
            
            % GDOP
            Q = inv(G' * G);
            gdop = sqrt(trace(Q(1:3, 1:3)));
            
            fprintf('GDOP: %.4f\n', gdop);
            obj.metrics.GDOP = gdop;
        end
        
        function [trajectory, covariance] = trackKalman(obj, measurements, dt)
            % 卡尔曼滤波跟踪
            % measurements: Tx3矩阵，每行是测量位置
            
            fprintf('卡尔曼滤波跟踪中...\n');
            
            T = size(measurements, 1);
            
            % 状态：[x, y, z, vx, vy, vz]
            F = [eye(3), dt*eye(3);
                 zeros(3), eye(3)];
            
            H = [eye(3), zeros(3)];
            
            % 噪声协方差
            Q = diag([0.1, 0.1, 0.1, 1, 1, 1]);
            R = 5^2 * eye(3);
            
            % 初始化
            x = [measurements(1, :)'; zeros(3, 1)];
            P = diag([10, 10, 10, 5, 5, 5]);
            
            trajectory = zeros(6, T);
            covariance = zeros(6, 6, T);
            
            for t = 1:T
                % 预测
                x_pred = F * x;
                P_pred = F * P * F' + Q;
                
                % 更新
                y = measurements(t, :)' - H * x_pred;
                S = H * P_pred * H' + R;
                K = P_pred * H' / S;
                
                x = x_pred + K * y;
                P = (eye(6) - K * H) * P_pred;
                
                trajectory(:, t) = x;
                covariance(:, :, t) = P;
            end
            
            fprintf('  跟踪完成！\n');
        end
        
        function [trajectory, weights] = trackParticle(obj, measurements, num_particles)
            % 粒子滤波跟踪
            fprintf('粒子滤波跟踪中（粒子数：%d）...\n', num_particles);
            
            T = size(measurements, 1);
            
            % 初始化粒子
            particles = zeros(6, num_particles);  % [x,y,z,vx,vy,vz]
            particles(1:3, :) = repmat(measurements(1, :)', 1, num_particles) + ...
                                randn(3, num_particles) * 5;
            weights = ones(1, num_particles) / num_particles;
            
            trajectory = zeros(6, T);
            
            for t = 1:T
                % 预测（运动模型）
                dt = 1.0;
                particles(1:3, :) = particles(1:3, :) + particles(4:6, :) * dt;
                particles = particles + randn(size(particles)) * [1;1;1;0.1;0.1;0.1];
                
                % 更新权重（测量模型）
                for p = 1:num_particles
                    predicted_pos = particles(1:3, p);
                    innovation = measurements(t, :)' - predicted_pos;
                    likelihood = exp(-0.5 * innovation' * innovation / 25);
                    weights(p) = weights(p) * likelihood;
                end
                
                % 归一化权重
                weights = weights / sum(weights);
                
                % 重采样（如果需要）
                Neff = 1 / sum(weights.^2);
                if Neff < num_particles / 2
                    % 系统重采样（不依赖 Statistics Toolbox 的 randsample）
                    cdf = cumsum(weights);
                    u = (rand() + (0:num_particles-1)) / num_particles;
                    indices = zeros(1, num_particles);
                    j = 1;
                    for ii = 1:num_particles
                        while u(ii) > cdf(j)
                            j = j + 1;
                        end
                        indices(ii) = j;
                    end
                    particles = particles(:, indices);
                    weights = ones(1, num_particles) / num_particles;
                end
                
                % 状态估计（加权平均）
                trajectory(:, t) = particles * weights';
            end
            
            fprintf('  跟踪完成！\n');
        end
        
        function visualizeTracking(obj, true_traj, estimated_traj, title_str)
            % 可视化跟踪结果
            figure('Color', 'w');
            
            % 3D轨迹
            subplot(2,2,[1,3]);
            plot3(true_traj(1,:), true_traj(2,:), true_traj(3,:), ...
                  'g-', 'LineWidth', 2); hold on;
            plot3(estimated_traj(1,:), estimated_traj(2,:), estimated_traj(3,:), ...
                  'b--', 'LineWidth', 2);
            
            % 绘制基站
            scatter3(obj.base_stations(:,1), obj.base_stations(:,2), ...
                    obj.base_stations(:,3), 200, 'r^', 'filled');
            
            legend('真实轨迹', '估计轨迹', '基站');
            xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
            title(title_str, 'FontSize', 14);
            grid on; axis equal;
            
            % X-Y平面投影
            subplot(2,2,2);
            plot(true_traj(1,:), true_traj(2,:), 'g-', 'LineWidth', 2); hold on;
            plot(estimated_traj(1,:), estimated_traj(2,:), 'b--', 'LineWidth', 2);
            scatter(obj.base_stations(:,1), obj.base_stations(:,2), 100, 'r^', 'filled');
            xlabel('X (m)'); ylabel('Y (m)');
            title('XY平面投影');
            grid on; axis equal; legend('真实', '估计', '基站');
            
            % 误差曲线
            subplot(2,2,4);
            errors = sqrt(sum((true_traj(1:3,:) - estimated_traj(1:3,:)).^2, 1));
            plot(errors, 'LineWidth', 2);
            xlabel('时间步'); ylabel('定位误差 (m)');
            title(sprintf('RMSE: %.2f m', sqrt(mean(errors.^2))));
            grid on;
        end
    end
end

%% 使用示例
function demo_localization()
    % 创建基站布局（四面体）
    stations = [
        0,    0,    0;
        1000, 0,    0;
        500,  866,  0;
        500,  289,  816
    ];
    
    % 创建定位系统
    loc_sys = MultiStationLocalization(stations);
    
    %% 测试TDOA定位
    true_pos = [600; 400; 300];
    c = 3e8;
    
    % 生成TDOA测量
    d1 = norm(true_pos - stations(1,:)');
    time_diffs = zeros(3, 1);
    for i = 2:4
        di = norm(true_pos - stations(i,:)');
        time_diffs(i-1) = (di - d1) / c + randn() * 1e-9;
    end
    
    est_pos = loc_sys.localizeTDOA(time_diffs, c);
    fprintf('TDOA误差: %.2f m\n', norm(est_pos - true_pos));
    
    %% 测试卡尔曼跟踪
    T = 100;
    dt = 0.1;
    true_trajectory = zeros(3, T);
    
    % 生成真实轨迹
    v = [10; 5; 2];  % 速度
    for t = 1:T
        true_trajectory(:, t) = true_pos + v * (t * dt);
    end
    
    % 添加测量噪声
    measurements = true_trajectory' + randn(T, 3) * 5;
    
    % 卡尔曼跟踪
    [traj_kf, ~] = loc_sys.trackKalman(measurements, dt);
    
    % 可视化
    loc_sys.visualizeTracking(true_trajectory, traj_kf(1:3,:), ...
                              '卡尔曼滤波跟踪');
end
