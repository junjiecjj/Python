%% 生成阵列接收信号
function rx_array_signal = received_array(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)
    % 参数设置
    % St LFM发送信号
    % array_phase 阵列相位
    % N_R 阵元数
    % num_pulses 脉冲数
    % fd 多普勒频移
    % PRF 脉冲重复频率
    % td 目标延迟时间
    % N_PRT 单个PRT的采样点数
    % N_d 目标时延的采样点数
    % N_st 单个脉冲内的采样点数
    % 返回阵列接收信号 （脉冲数×单个PRT的采样点数，阵元数）

    % 初始化多通道接收信号    
    rx_array_signal = zeros(num_pulses * N_PRT, N_R);
    for n = 0:num_pulses-1
        % 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = exp(1j * 2 * pi * fd * (n / PRF + td));
        % 生成阵列接收信号
        for k = 1:N_R
            % 每个阵元的相位补偿
            R_phase = array_phase(k) * doppler_phase;
            % 计算信号位置
            start_idx = n * N_PRT + N_d;
            end_idx = start_idx + N_st;
            % 截断处理
            if end_idx > num_pulses * N_PRT
                end_idx = num_pulses * N_PRT;
                valid_len = end_idx - start_idx;
                rx_array_signal(start_idx+1:start_idx+valid_len, k) = ...
                    rx_array_signal(start_idx+1:start_idx+valid_len, k) + ...
                    St(1:valid_len).' * R_phase;
            else
                rx_array_signal(start_idx+1:end_idx, k) = ...
                    rx_array_signal(start_idx+1:end_idx, k) + ...
                    St.' * R_phase;
            end
        end
    end
end