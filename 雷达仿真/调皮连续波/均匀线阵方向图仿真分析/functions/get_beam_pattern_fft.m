function [p_dB, angle_bins] = get_beam_pattern_fft(ula, coefficient, fft_size)
% GET_BEAM_PATTERN_FFT 计算波束方向图(FFT方法)
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 使用FFT方法计算均匀线阵的波束方向图
% 版本: 1.0
% 输入参数:
%   ula - 均匀线阵结构体
%   coefficient - 加权系数向量
%   fft_size    - FFT点数，默认256
% 输出参数:
%   p_dB      - 波束方向图 [dB]
%   angle_bins - 角度采样点 [rad]

    if nargin < 3
        fft_size = 256;
    end
    
    C = fftshift(fft(coefficient, fft_size));
    p_dB = 20 * log10(abs(C));
    p_dB = p_dB - max(p_dB);
    
    angle_bins = get_angle_bins(ula, ula.d, fft_size);
end