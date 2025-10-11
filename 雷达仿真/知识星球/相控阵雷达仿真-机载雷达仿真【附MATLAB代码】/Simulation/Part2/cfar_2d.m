%% 二维CA-CFAR检测函数
function [detection_map, target_info] = cfar_2d(input_data, guard_win, train_win, P_fa, range_bins, speed_bins, R_true, v_true)
    % input_data: 输入数据矩阵（距离门×速度门）
    % guard_win: 保护单元 [距离保护, 多普勒保护]
    % train_win: 参考单元 [距离参考, 多普勒参考]
    % P_fa: 虚警概率
    % range_bins: 距离轴向量
    % speed_bins: 速度轴向量
    % R_true: 真实目标距离
    % v_true: 真实目标速度
    
    [num_range, num_doppler] = size(input_data);
    detection_map = zeros(num_range, num_doppler);
    
    % 计算阈值因子
    num_ref_cells = (2*train_win(1)+2*guard_win(1)+1)*(2*train_win(2)+2*guard_win(2)+1) - ...
                   (2*guard_win(1)+1)*(2*guard_win(2)+1);
    alpha = num_ref_cells*(P_fa^(-1/num_ref_cells) - 1);
    
    % 滑动窗口检测
    for range_idx = 1+train_win(1)+guard_win(1) : num_range-train_win(1)-guard_win(1)
        for doppler_idx = 1+train_win(2)+guard_win(2) : num_doppler-train_win(2)-guard_win(2)
            % 定义检测区域
            range_win = range_idx-train_win(1)-guard_win(1):range_idx+train_win(1)+guard_win(1);
            doppler_win = doppler_idx-train_win(2)-guard_win(2):doppler_idx+train_win(2)+guard_win(2);
            
            % 提取参考单元
            ref_cells = input_data(range_win, doppler_win);
            
            % 去除保护单元
            ref_cells(train_win(1)+1:end-train_win(1), train_win(2)+1:end-train_win(2)) = NaN;
            
            % 计算噪声基底
            noise_level = mean(ref_cells(:), "omitmissing");
            threshold = alpha * noise_level;
            
            % 检测判决
            if input_data(range_idx, doppler_idx) > threshold
                detection_map(range_idx, doppler_idx) = 1;
            end
        end
    end
        
    % 匹配真实目标
    [~, true_range_idx] = min(abs(range_bins - R_true/1e3));
    [~, true_speed_idx] = min(abs(speed_bins - v_true));

    % 判断真实目标是否被检测到
    is_detected = false;
    if detection_map(true_range_idx, true_speed_idx) == 1
        detected_range = range_bins(true_range_idx);
        detected_speed = speed_bins(true_speed_idx);
        is_detected = true;
    end
    
    if ~is_detected
        [detected_ranges, detected_speeds] = find(detection_map == 1);
        min_dist = inf;
        detected_range = NaN;
        detected_speed = NaN;
        for k = 1:length(detected_ranges)
            current_dist = sqrt(...
                (range_bins(detected_ranges(k)) - R_true/1e3)^2 + ...
                (speed_bins(detected_speeds(k)) - v_true)^2);
            if current_dist < min_dist
                min_dist = current_dist;
                detected_range = range_bins(detected_ranges(k));
                detected_speed = speed_bins(detected_speeds(k));
            end
        end
    end
    
    % ==== 错误处理：未检测到目标时给出警告 ====
    if isnan(detected_range)
        warning('未检测到真实目标，请调整CFAR参数!');
    end
    
    % 计算误差
    if ~isnan(detected_range)
        range_error = abs(detected_range - R_true/1e3)/(R_true/1e3)*100;
        speed_error = abs(detected_speed - v_true)/abs(v_true)*100;
    else
        range_error = NaN;
        speed_error = NaN;
    end
    
    % 输出结果
    target_info = struct(...
        'TrueRange', R_true/1e3, ...
        'TrueSpeed', v_true, ...
        'DetectedRange', detected_range, ...
        'DetectedSpeed', detected_speed, ...
        'RangeError', range_error, ...
        'SpeedError', speed_error);
end
