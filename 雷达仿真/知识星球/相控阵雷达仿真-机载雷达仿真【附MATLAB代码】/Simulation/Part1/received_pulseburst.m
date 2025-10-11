%% 生成脉冲串接收信号
function received = received_pulseburst(St, num_pulses, fd, PRF, td, N_PRT, N_d, N_st)
    % 参数设置
    % St LFM发送信号
    % num_pulses 脉冲数
    % fd 多普勒频移
    % PRF 脉冲重复频率
    % td 目标延迟时间
    % N_PRT 单个PRT的采样点数
    % N_d 目标时延的采样点数
    % N_st 单个脉冲内的采样点数
    % 返回脉冲串接收信号 （脉冲数×单个PRT的采样点数，1）

    % 生成脉冲串接收信号
    received = zeros(num_pulses * N_PRT, 1);
    for n = 0:num_pulses-1
        % 生成单脉冲回波（含多普勒相位和时间延迟）
        doppler_phase = exp(1j * 2 * pi * fd * (n / PRF + td));
        % 计算回波在接收窗口中的位置
        start_idx = n * N_PRT + N_d;
        end_idx = start_idx + N_st;
    
        % 截断处理防止越界
        if end_idx > num_pulses * N_PRT
            end_idx = num_pulses * N_PRT;
            valid_len = end_idx - start_idx;
            received(start_idx+1:end_idx) = received(start_idx+1:end_idx) + ...
                St(1:valid_len).' * doppler_phase;
        else
            received(start_idx+1:end_idx) = received(start_idx+1:end_idx) + ...
                St.' * doppler_phase;
        end
    end
end