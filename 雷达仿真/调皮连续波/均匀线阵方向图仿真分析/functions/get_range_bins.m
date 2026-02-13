function range_bins = get_range_bins(cc)
% GET_RANGE_BINS 获取距离采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据Chirp配置计算距离采样点
% 版本: 1.0
% 输入参数:
%   cc - Chirp配置结构体
% 输出参数:
%   range_bins - 距离采样点 [m]

    sz = get_fft_size(cc.num_adc);
    if cc.num_adc == sz
        range_bins = 0:cc.range_resolution:(cc.range_max-cc.range_resolution);
    else
        range_bins = linspace(0, cc.range_max, sz);
    end
end