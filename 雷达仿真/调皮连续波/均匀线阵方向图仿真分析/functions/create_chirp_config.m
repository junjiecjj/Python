function config = create_chirp_config()
%CREATE_CHIRP_CONFIG 创建Chirp配置结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含FMCW雷达Chirp参数的配置结构体
% 版本: 1.0
% 输出参数:
%   config - Chirp配置结构体，包含:
%            .speed_of_light   - 光速 [m/s]
%            .start_freq       - 起始频率 [GHz]
%            .chirp_duration   - Chirp时长 [us]
%            .num_adc          - ADC采样数
%            .num_chirps       - Chirp数量
%            .chirp_slope      - Chirp斜率 [MHz/us]
%            .start_lambda     - 起始波长 [m]
%            .sampling_rate    - 采样率 [MHz]
%            .sampling_delta   - 采样间隔 [us]
%            .bandwidth        - 信号带宽 [MHz]
%            .chirp_timesteps  - Chirp时间序列 [us]
%            .range_max        - 最大探测距离 [m]
%            .range_resolution - 距离分辨率 [m]
%            .doppler_max      - 最大多普勒速度 [m/s]

config.speed_of_light = 299792458.0;
config.start_freq = 76.8;
config.chirp_duration = 50;
config.num_adc = 250;
config.num_chirps = 128;
config.chirp_slope = 30;
config = update_chirp_config(config);
end
