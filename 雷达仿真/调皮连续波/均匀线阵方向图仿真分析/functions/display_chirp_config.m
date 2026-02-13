function display_chirp_config(config)
%DISPLAY_CHIRP_CONFIG 显示Chirp配置参数信息
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 在命令行窗口打印Chirp配置参数信息
% 版本: 1.0
% 输入参数:
%   config - Chirp配置结构体

    fprintf('\n========== Chirp配置参数 ==========\n');
    fprintf('基本参数:\n');
    fprintf('  起始频率:     %.2f GHz\n', config.start_freq);
    fprintf('  Chirp时长:    %.1f us\n', config.chirp_duration);
    fprintf('  ADC采样数:    %d\n', config.num_adc);
    fprintf('  Chirp数量:    %d\n', config.num_chirps);
    fprintf('  Chirp斜率:    %.1f MHz/us\n', config.chirp_slope);
    fprintf('\n计算参数:\n');
    fprintf('  波长:         %.4f mm\n', config.start_lambda * 1000);
    fprintf('  采样率:       %.2f MHz\n', config.sampling_rate);
    fprintf('  带宽:         %.1f MHz\n', config.bandwidth);
    fprintf('\n性能指标:\n');
    fprintf('  最大距离:     %.2f m\n', config.range_max);
    fprintf('  距离分辨率:   %.3f m\n', config.range_resolution);
    fprintf('  最大速度:     %.2f m/s\n', config.doppler_max);
    fprintf('====================================\n\n');
end
