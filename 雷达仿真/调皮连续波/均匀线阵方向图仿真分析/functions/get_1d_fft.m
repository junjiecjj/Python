function rfft = get_1d_fft(data_cube)
% GET_1D_FFT 一维FFT处理
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 对雷达数据立方体进行距离维FFT处理
% 版本: 1.0
% 输入参数:
%   data_cube - 数据立方体 [... × ADC]
% 输出参数:
%   rfft - 距离维FFT结果

    sz = get_fft_size(size(data_cube, end));
    rfft = fft(data_cube, sz, size(data_cube, ndims(data_cube)));
end