function config = update_chirp_config(config)
%UPDATE_CHIRP_CONFIG 更新Chirp配置结构体的计算参数
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据基本参数计算Chirp配置的派生参数
% 版本: 1.0
% 输入参数:
%   config - Chirp配置结构体(包含基本参数)
% 输出参数:
%   config - 更新后的Chirp配置结构体(包含派生参数)

    config.start_lambda = config.speed_of_light / (config.start_freq * 1e9);
    config.sampling_rate = config.num_adc / config.chirp_duration;
    config.sampling_delta = 1 / config.sampling_rate;
    config.bandwidth = config.chirp_duration * config.chirp_slope;
    
    num_steps = config.num_adc;
    config.chirp_timesteps = (0:num_steps - 1) * config.sampling_delta;
    
    config.range_max = config.sampling_rate * config.speed_of_light ...
        / (config.chirp_slope * 1e6) / 2;
    config.range_resolution = config.speed_of_light ...
        / (config.bandwidth * 1e6) / 2;
    config.doppler_max = config.start_lambda ...
        / (config.chirp_duration * 1e-6) / 4;
end
