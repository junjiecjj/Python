function angle_bins = get_angle_bins(ula, d, fft_size)
% GET_ANGLE_BINS 获取角度采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算基于FFT的角度采样点
% 版本: 1.0
% 输入参数:
%   ula - 均匀线阵结构体
%   d        - 归一化间距
%   fft_size - FFT点数
% 输出参数:
%   angle_bins - 角度采样点 [rad]

    if nargin < 3
        fft_size = 256;
    end
    
    radrange = -pi : 2*pi/fft_size : pi-2*pi/fft_size;
    angle_bins = asin(radrange / (2 * pi * d));
end