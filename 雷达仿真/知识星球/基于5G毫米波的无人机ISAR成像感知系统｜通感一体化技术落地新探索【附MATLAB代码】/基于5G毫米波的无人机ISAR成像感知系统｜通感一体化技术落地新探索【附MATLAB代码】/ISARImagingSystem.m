%% 完整ISAR成像系统 - 研究级实现
% =============================================
% 项目：基于5G基站的无人机成像感知
% 研究方向：定位导航与智能感知技术
% 
% 功能：
% 1. 多目标场景仿真
% 2. 运动补偿（包括平动和转动）
% 3. 自聚焦算法（最小熵、对比度）
% 4. 成像质量评估
% =============================================

classdef ISARImagingSystem < handle
    % ISAR成像系统类
    
    properties
        % 雷达参数
        fc          % 载波频率
        B           % 带宽
        Tp          % 脉冲宽度
        PRF         % 脉冲重复频率
        c = 3e8     % 光速
        
        % 观测参数
        T_obs       % 观测时间
        R0          % 初始距离
        
        % 数据存储
        echo_data           % 原始回波
        range_compressed    % 距离压缩后
        motion_compensated  % 运动补偿后
        isar_image          % 最终ISAR图像
        
        % 性能指标
        metrics
    end
    
    methods
        function obj = ISARImagingSystem(config)
            % 构造函数
            obj.fc = config.fc;
            obj.B = config.B;
            obj.Tp = config.Tp;
            obj.PRF = config.PRF;
            obj.T_obs = config.T_obs;
            obj.R0 = config.R0;
            
            % 初始化性能指标
            obj.metrics = struct();
        end
        
        function echo = simulateMultiTarget(obj, targets, motion_params)
            % 多目标场景仿真
            % targets: Nx3矩阵，每行是[x, y, z, RCS]
            % motion_params: 运动参数结构体
            
            fprintf('正在仿真多目标场景...\n');
            
            % 时间轴
            slow_time = 0:1/obj.PRF:obj.T_obs;
            fast_time = 0:1/(2*obj.B):obj.Tp;
            num_pulses = length(slow_time);
            num_samples = length(fast_time);
            
            % 初始化回波
            echo = zeros(num_samples, num_pulses);
            Kr = obj.B / obj.Tp;
            
            % 生成每个目标的回波
            for t_idx = 1:size(targets, 1)
                target = targets(t_idx, :);
                x0 = target(1); y0 = target(2); z0 = target(3);
                RCS = target(4);  % 雷达散射截面
                
                for p = 1:num_pulses
                    t = slow_time(p);
                    
                    % 目标运动模型
                    [x, y, z] = obj.targetMotion(x0, y0, z0, t, motion_params);
                    
                    % 计算瞬时距离
                    R = sqrt(x^2 + y^2 + z^2);
                    tau = 2*R / obj.c;
                    
                    % 幅度因子（考虑RCS和距离衰减）
                    amplitude = sqrt(RCS) / R^2;
                    
                    % 生成LFM信号
                    sig = amplitude * exp(1j*pi*Kr*(fast_time - tau).^2) .* ...
                          exp(-1j*4*pi*obj.fc*R/obj.c);
                    sig(fast_time < tau) = 0;
                    
                    echo(:, p) = echo(:, p) + sig';
                end
            end
            
            % 添加系统噪声
            SNR_dB = 15;
            try
                echo = awgn(echo, SNR_dB, 'measured');
            catch
                signal_power = mean(abs(echo(:)).^2);
                noise_power = signal_power / (10^(SNR_dB/10));
                noise = sqrt(noise_power/2) * (randn(size(echo)) + 1j*randn(size(echo)));
                echo = echo + noise;
            end
            
            obj.echo_data = echo;
            fprintf('  多目标回波生成完成！\n');
        end
        
        function [x, y, z] = targetMotion(obj, x0, y0, z0, t, params)
            % 目标运动模型
            % params包含：velocity（速度向量），omega（角速度），jitter（抖动）
            
            % 平动
            x = x0 + params.velocity(1) * t;
            y = y0 + params.velocity(2) * t;
            z = z0 + params.velocity(3) * t;
            
            % 旋转（绕z轴）
            if isfield(params, 'omega')
                theta = params.omega * t;
                x_rot = x * cos(theta) - y * sin(theta);
                y_rot = x * sin(theta) + y * cos(theta);
                x = x_rot;
                y = y_rot;
            end
            
            % 添加微动（模拟风扰、发动机振动等）
            if isfield(params, 'jitter')
                x = x + params.jitter * randn();
                y = y + params.jitter * randn();
                z = z + params.jitter * randn();
            end
        end
        
        function rc_data = rangeCompression(obj, echo)
            % 距离压缩（脉冲压缩）
            fprintf('正在进行距离压缩...\n');
            
            fast_time = 0:1/(2*obj.B):obj.Tp;
            Kr = obj.B / obj.Tp;
            
            % 匹配滤波器
            ref_signal = exp(1j*pi*Kr*fast_time.^2);
            matched_filter = conj(fliplr(ref_signal));
            
            % 对每个脉冲进行压缩
            [num_samples, num_pulses] = size(echo);
            rc_data = zeros(size(echo));
            
            for p = 1:num_pulses
                rc_data(:, p) = conv(echo(:, p), matched_filter, 'same');
            end
            
            obj.range_compressed = rc_data;
            
            % 计算距离分辨率
            range_res = obj.c / (2 * obj.B);
            fprintf('  距离分辨率: %.2f cm\n', range_res * 100);
        end
        
        function mc_data = motionCompensation(obj, rc_data)
            % 运动补偿（包括包络对齐和相位校正）
            fprintf('正在进行运动补偿...\n');
            
            % 1. 包络对齐（Envelope Alignment）
            [num_samples, num_pulses] = size(rc_data);
            aligned_data = zeros(size(rc_data));
            
            % 选择参考脉冲（通常选中间脉冲）
            ref_pulse = round(num_pulses / 2);
            ref_profile = abs(rc_data(:, ref_pulse));
            
            for p = 1:num_pulses
                current_profile = abs(rc_data(:, p));
                
                % 互相关寻找最大值位置
                [corr_val, lags] = xcorr(current_profile, ref_profile);
                [~, max_idx] = max(abs(corr_val));
                shift = lags(max_idx);
                
                % 循环移位对齐
                aligned_data(:, p) = circshift(rc_data(:, p), -shift);
            end
            
            % 2. 相位校正（Phase Correction）
            % 提取强散射点
            power_profile = mean(abs(aligned_data).^2, 2);
            [~, dominant_bins] = sort(power_profile, 'descend');
            strong_scatterers = dominant_bins(1:min(5, length(dominant_bins)));
            
            % 估计相位误差
            phase_error = zeros(1, num_pulses);
            for p = 1:num_pulses
                % 使用强散射点的相位平均
                phases = angle(aligned_data(strong_scatterers, p));
                phase_error(p) = mean(phases);
            end
            
            % 去除线性相位（对应平动）
            t = 1:num_pulses;
            p_fit = polyfit(t, phase_error, 1);
            phase_linear = polyval(p_fit, t);
            phase_error_corrected = phase_error - phase_linear;
            
            % 应用相位校正
            mc_data = aligned_data .* exp(-1j * phase_error_corrected);
            
            obj.motion_compensated = mc_data;
            fprintf('  运动补偿完成！\n');
        end
        
        function img = autofocus(obj, data, method)
            % 自聚焦算法
            % method: 'entropy' | 'contrast' | 'pga'
            
            fprintf('正在进行自聚焦（方法：%s）...\n', method);
            
            switch method
                case 'entropy'
                    img = obj.autofocusEntropy(data);
                case 'contrast'
                    img = obj.autofocusContrast(data);
                case 'pga'
                    img = obj.autofocusPGA(data);
                otherwise
                    error('未知的自聚焦方法！');
            end
            
            obj.isar_image = img;
        end
        
        function img = autofocusEntropy(obj, data)
            % 最小熵自聚焦
            
            % 定义优化目标
            objective = @(params) obj.calculateEntropy(data, params);
            
            % PSO优化
            nvars = 2;  % [一次相位系数, 二次相位系数]
            lb = [-0.1, -0.01];
            ub = [0.1, 0.01];
            
            try
                options = optimoptions('particleswarm', ...
                    'SwarmSize', 30, ...
                    'MaxIterations', 50, ...
                    'Display', 'iter');
                [best_params, best_entropy] = particleswarm(objective, nvars, lb, ub, options);
            catch
                % 无 Global Optimization Toolbox，用 fminsearch 替代
                x0 = (lb + ub) / 2;
                opts = optimset('MaxIter', 200, 'Display', 'off');
                [best_params, best_entropy] = fminsearch(objective, x0, opts);
            end

            % 应用最优参数
            img = obj.applyPhaseCorrection(data, best_params);

            fprintf('  最优熵值: %.4f\n', best_entropy);
            obj.metrics.entropy = best_entropy;
            obj.metrics.entropy_params = best_params;
        end
        
        function img = autofocusContrast(obj, data)
            % 对比度自聚焦
            
            objective = @(params) -obj.calculateContrast(data, params);
            
            nvars = 2;
            lb = [-0.1, -0.01];
            ub = [0.1, 0.01];
            
            try
                options = optimoptions('particleswarm', ...
                    'SwarmSize', 30, ...
                    'MaxIterations', 50, ...
                    'Display', 'off');
                [best_params, neg_contrast] = particleswarm(objective, nvars, lb, ub, options);
            catch
                x0 = (lb + ub) / 2;
                opts = optimset('MaxIter', 200, 'Display', 'off');
                [best_params, neg_contrast] = fminsearch(objective, x0, opts);
            end
            
            img = obj.applyPhaseCorrection(data, best_params);
            
            fprintf('  对比度: %.4f\n', -neg_contrast);
            obj.metrics.contrast = -neg_contrast;
        end
        
        function img = autofocusPGA(obj, data)
            % 相位梯度自聚焦（Phase Gradient Autofocus）
            
            max_iterations = 20;
            [num_samples, num_pulses] = size(data);
            corrected_data = data;
            
            for iter = 1:max_iterations
                % 当前图像
                current_img = fftshift(fft(corrected_data, [], 2), 2);
                
                % 选择强散射点
                power_map = abs(current_img).^2;
                threshold = 0.8 * max(power_map(:));
                [rows, cols] = find(power_map > threshold);
                
                if isempty(rows)
                    break;
                end
                
                % 估计相位误差梯度
                phase_gradient = zeros(1, num_pulses);
                for i = 1:length(rows)
                    profile = corrected_data(rows(i), :);
                    phase = angle(profile);
                    phase_gradient = phase_gradient + diff([phase, phase(end)]);
                end
                phase_gradient = phase_gradient / length(rows);
                
                % 积分得到相位误差
                phase_error = cumsum(phase_gradient);
                
                % 应用校正
                corrected_data = corrected_data .* exp(-1j * phase_error);
                
                % 检查收敛
                if iter > 1 && norm(phase_error) < 1e-3
                    break;
                end
            end
            
            img = fftshift(fft(corrected_data, [], 2), 2);
            fprintf('  PGA迭代次数: %d\n', iter);
        end
        
        function entropy = calculateEntropy(obj, data, params)
            % 计算图像熵
            img = obj.applyPhaseCorrection(data, params);
            img_abs = abs(img) / sum(abs(img(:)));
            img_abs(img_abs == 0) = eps;
            entropy = -sum(img_abs(:) .* log(img_abs(:)));
        end
        
        function contrast = calculateContrast(obj, data, params)
            % 计算图像对比度
            img = obj.applyPhaseCorrection(data, params);
            img_abs = abs(img);
            contrast = std(img_abs(:)) / mean(img_abs(:));
        end
        
        function img = applyPhaseCorrection(obj, data, params)
            % 应用相位校正
            num_pulses = size(data, 2);
            t = 1:num_pulses;
            phase_corr = params(1) * t + params(2) * t.^2;
            corrected = data .* exp(-1j * phase_corr);
            img = fftshift(fft(corrected, [], 2), 2);
        end
        
        function evaluateImageQuality(obj)
            % 评估成像质量
            fprintf('\n===== 成像质量评估 =====\n');
            
            img = abs(obj.isar_image);
            
            % 1. 信噪比（SNR）
            signal = max(img(:));
            noise = mean(img(img < 0.1*max(img(:))));
            SNR = 20 * log10(signal / noise);
            fprintf('SNR: %.2f dB\n', SNR);
            obj.metrics.SNR = SNR;
            
            % 2. 图像对比度
            contrast = std(img(:)) / mean(img(:));
            fprintf('对比度: %.4f\n', contrast);
            obj.metrics.contrast = contrast;
            
            % 3. 图像熵
            img_norm = img / sum(img(:));
            img_norm(img_norm == 0) = eps;
            entropy = -sum(img_norm(:) .* log(img_norm(:)));
            fprintf('熵: %.4f\n', entropy);
            obj.metrics.entropy = entropy;
            
            % 4. 峰值旁瓣比（PSLR）
            [max_val, max_idx] = max(img(:));
            img_copy = img;
            img_copy(max_idx) = 0;
            sidelobe_max = max(img_copy(:));
            PSLR = 20 * log10(max_val / sidelobe_max);
            fprintf('峰值旁瓣比: %.2f dB\n', PSLR);
            obj.metrics.PSLR = PSLR;
            
            % 5. 分辨率估计
            range_res = obj.c / (2 * obj.B);
            fprintf('距离分辨率: %.2f cm\n', range_res * 100);
            obj.metrics.range_resolution = range_res;
            
            fprintf('========================\n\n');
        end
        
        function visualize(obj, save_path)
            % 可视化结果
            figure('Position', [50, 50, 1600, 1000], 'Color', 'w');
            
            % 子图1：原始回波
            subplot(2,3,1);
            imagesc(abs(obj.echo_data));
            colormap(jet); colorbar;
            title('原始回波', 'FontSize', 14);
            xlabel('慢时间'); ylabel('快时间');
            
            % 子图2：距离压缩后
            subplot(2,3,2);
            imagesc(abs(obj.range_compressed));
            colorbar;
            title('距离压缩', 'FontSize', 14);
            xlabel('慢时间'); ylabel('距离单元');
            
            % 子图3：运动补偿后
            subplot(2,3,3);
            imagesc(abs(obj.motion_compensated));
            colorbar;
            title('运动补偿', 'FontSize', 14);
            xlabel('慢时间'); ylabel('距离单元');
            
            % 子图4：ISAR图像（线性尺度）
            subplot(2,3,4);
            imagesc(abs(obj.isar_image));
            colorbar;
            title('ISAR图像（线性）', 'FontSize', 14);
            xlabel('多普勒'); ylabel('距离');
            
            % 子图5：ISAR图像（dB尺度，归一化）
            subplot(2,3,5);
            imagesc(20*log10(abs(obj.isar_image)/max(abs(obj.isar_image(:))) + eps));
            colorbar;
            caxis([-40, 0]);
            title('ISAR图像（dB）', 'FontSize', 14);
            xlabel('多普勒'); ylabel('距离');
            
            % 子图6：性能指标
            subplot(2,3,6);
            axis off;
            text_str = sprintf(['成像性能指标\n\n' ...
                'SNR: %.2f dB\n' ...
                '对比度: %.4f\n' ...
                '熵: %.4f\n' ...
                'PSLR: %.2f dB\n' ...
                '距离分辨率: %.2f cm'], ...
                obj.metrics.SNR, ...
                obj.metrics.contrast, ...
                obj.metrics.entropy, ...
                obj.metrics.PSLR, ...
                obj.metrics.range_resolution * 100);
            text(0.1, 0.5, text_str, 'FontSize', 12, ...
                'VerticalAlignment', 'middle');
            
            if nargin > 1
                saveas(gcf, save_path);
                fprintf('图像已保存: %s\n', save_path);
            end
        end
    end
end

%% 使用示例
function demo_usage()
    % 配置参数
    config = struct();
    config.fc = 28e9;
    config.B = 400e6;
    config.Tp = 1e-6;
    config.PRF = 1000;
    config.T_obs = 2;
    config.R0 = 1000;
    
    % 创建系统实例
    system = ISARImagingSystem(config);
    
    % 定义多目标场景
    % 格式：[x, y, z, RCS]
    targets = [
        0,    0,    0,   1.0;   % 中心目标
        0.3,  0,    0,   0.5;   % 右侧目标
       -0.3,  0,    0,   0.5;   % 左侧目标
        0,    0.25, 0,   0.3    % 前方目标
    ];
    
    % 运动参数
    motion = struct();
    motion.velocity = [5, 0, 0];  % 速度 5m/s
    motion.omega = 0.5;            % 旋转 0.5rad/s
    motion.jitter = 0.01;          % 微动 1cm
    
    % 仿真
    system.simulateMultiTarget(targets, motion);
    
    % 处理
    system.rangeCompression(system.echo_data);
    system.motionCompensation(system.range_compressed);
    system.autofocus(system.motion_compensated, 'entropy');
    
    % 评估
    system.evaluateImageQuality();
    
    % 可视化
    system.visualize('isar_result_advanced.png');
end
