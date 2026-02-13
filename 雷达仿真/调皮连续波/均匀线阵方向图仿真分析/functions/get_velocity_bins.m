function velocity_bins = get_velocity_bins(cc)
% GET_VELOCITY_BINS 获取速度采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据Chirp配置计算速度采样点
% 版本: 1.0
% 输入参数:
%   cc - Chirp配置结构体
% 输出参数:
%   velocity_bins - 速度采样点 [m/s]

    sz = get_fft_size(cc.num_chirps);
    velocity_bins = linspace(-cc.doppler_max, cc.doppler_max, sz);
end