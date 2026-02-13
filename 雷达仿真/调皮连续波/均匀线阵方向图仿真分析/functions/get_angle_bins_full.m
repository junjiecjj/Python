function angle_bins = get_angle_bins_full(ula, fft_size)
% GET_ANGLE_BINS_FULL 获取完整角度范围采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算完整角度范围的采样点，范围[-π/2, π/2)
% 版本: 1.0
% 输入参数:
%   ula      - 均匀线阵结构体
%   fft_size - 采样点数，默认256
% 输出参数:
%   angle_bins - 角度采样点 [rad]，范围[-π/2, π/2)

    if nargin < 2
        fft_size = 256;
    end
    
    angle_bins = -pi/2 : pi/fft_size : pi/2 - pi/fft_size;
end