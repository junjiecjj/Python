function ts = get_timesteps(radar_sim, num_frames)
% GET_TIMESTEPS 生成时间序列
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 生成雷达信号的时间序列
% 版本: 1.0
% 输入参数:
%   radar_sim - 雷达仿真器结构体
%   num_frames - 帧数
% 输出参数:
%   ts - 时间序列 [frames × chirps × ADC]

    num_steps = radar_sim.cc.num_adc * radar_sim.cc.num_chirps * num_frames;
    end_time = radar_sim.cc.chirp_duration * radar_sim.cc.num_chirps * num_frames * 1e-6;
    ts = linspace(0, end_time, num_steps);
    ts = reshape(ts, [num_frames, radar_sim.cc.num_chirps, radar_sim.cc.num_adc]);
end