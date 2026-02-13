function fft_size = get_fft_size(sz)
% GET_FFT_SIZE 计算FFT点数(向上取整到2的幂次)
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算适合FFT处理的点数，向上取整到2的幂次
% 版本: 1.0
% 输入参数:
%   sz - 原始数据长度
% 输出参数:
%   fft_size - FFT点数

    if isempty(sz) || ~isnumeric(sz) || sz <= 0
        error('get_fft_size: 无效的输入参数 sz = %s', mat2str(sz));
    end
    fft_size = 2^ceil(log2(sz));
end