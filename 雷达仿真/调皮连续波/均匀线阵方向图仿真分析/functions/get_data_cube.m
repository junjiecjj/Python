function dc = get_data_cube(radar_sim, num_frames, targets)
% GET_DATA_CUBE 生成4D数据立方体
% 作者: Radar Engineer
% 时间: 2026-02-13
% 功能: 使用嵌套for循环生成包含复数中频信号的4D数据立方体
% 版本: 1.1
% 输入参数:
%   radar_sim - 雷达仿真器结构体
%   num_frames - 帧数
%   targets - 目标对象结构体元胞数组
% 输出参数:
%   dc - 数据立方体 [frames × TX × RX × chirps × ADC]

    num_adc = radar_sim.cc.num_adc;
    num_chirp = radar_sim.cc.num_chirps;
    num_tx = size(radar_sim.mc.txl, 1);
    num_rx = size(radar_sim.mc.rxl, 1);
    
    % Chirp参数
    slope = radar_sim.cc.chirp_slope * 1e12;  % [MHz/us] -> [Hz/s]
    fs = radar_sim.cc.start_freq * 1e9;       % [GHz] -> [Hz]
    speed_of_light = radar_sim.cc.speed_of_light;
    
    % 复相位函数
    complexPhase = @(tv) 2 * pi * (fs * tv + slope/2 * tv.^2);
    
    % Chirp时间序列 [1 × num_adc]
    ct = radar_sim.cc.chirp_timesteps(:)' * 1e-6;  % [us] -> [s], 确保是行向量
    
    % 初始化数据立方体 [frames × TX × RX × chirps × ADC]
    dc = zeros(num_frames, num_tx, num_rx, num_chirp, num_adc);
    
    % 获取完整时间序列
    ts = get_timesteps(radar_sim, num_frames);  % [frames × chirps × ADC]
    
    % 对每个目标生成信号
    for i = 1:length(targets)
        tg = targets{i};
        
        % 对每个帧
        for frame_idx = 1:num_frames
            % 对每个chirp
            for chirp_idx = 1:num_chirp
                % 对每个ADC采样
                for adc_idx = 1:num_adc
                    % 获取当前时间
                    current_time = ts(frame_idx, chirp_idx, adc_idx);
                    
                    % 计算目标在当前时间的位置
                    traj = get_target_trajectories_3d(tg, current_time);
                    
                    % 对每个TX
                    for tx_idx = 1:num_tx
                        % TX位置
                        tx_pos = radar_sim.mc.d_m * radar_sim.mc.txl(tx_idx, :)';
                        
                        % 对每个RX
                        for rx_idx = 1:num_rx
                            % RX位置
                            rx_pos = radar_sim.mc.d_m * radar_sim.mc.rxl(rx_idx, :)' + tx_pos;
                            
                            % 计算TX到目标的距离
                            tx_to_target = traj - tx_pos;
                            r_tx = vecnorm(tx_to_target);
                            
                            % 计算目标到RX的距离
                            target_to_rx = rx_pos - traj;
                            r_rx = vecnorm(target_to_rx);
                            
                            % 计算总距离和时延
                            total_distance = r_tx + r_rx;
                            delay = total_distance / speed_of_light;
                            
                            % 计算相位
                            ct_val = ct(adc_idx);
                            phase0 = complexPhase(ct_val);
                            phase = complexPhase(ct_val - delay);
                            
                            % 生成信号
                            signal = exp(1i * (phase0 - phase));
                            
                            % 叠加到数据立方体
                            dc(frame_idx, tx_idx, rx_idx, chirp_idx, adc_idx) = ...
                                dc(frame_idx, tx_idx, rx_idx, chirp_idx, adc_idx) + signal;
                        end % rx_idx
                    end % tx_idx
                end % adc_idx
            end % chirp_idx
        end % frame_idx
    end % target
end