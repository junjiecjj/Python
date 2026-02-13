function fft_sizes = get_fft_sizes(szl)
% GET_FFT_SIZES 计算多个FFT点数
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算多个维度的FFT点数
% 版本: 1.0
% 输入参数:
%   szl - 原始数据长度数组
% 输出参数:
%   fft_sizes - FFT点数元组

    fft_sizes = cell(1, length(szl));
    for i = 1:length(szl)
        fft_sizes{i} = get_fft_size(szl(i));
    end
end