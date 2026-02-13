function [p_dB, angle_bins] = get_beam_pattern_fft_2d(ura, coefficient, fft_size)
% GET_BEAM_PATTERN_FFT_2D 计算2D波束方向图
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 使用FFT方法计算均匀面阵的2D波束方向图
% 版本: 1.0
% 输入参数:
%   ura - 均匀面阵结构体
%   coefficient - 加权系数矩阵 [N_rows × N_cols]
%   fft_size    - FFT点数 [N_fft_el × N_fft_az]
% 输出参数:
%   p_dB      - 波束方向图 [dB]
%   angle_bins - 角度采样点 {theta_bins, phi_bins}

    if nargin < 3
        fft_size = [256, 256];
    end
    
    if isscalar(fft_size)
        fft_size = [fft_size, fft_size];
    end
    
    % 2D FFT
    C = fftshift(fft2(coefficient, fft_size(1), fft_size(2)));
    p_dB = 20 * log10(abs(C));
    p_dB = p_dB - max(p_dB(:));
    
    angle_bins = get_angle_bins_2d(ura, ura.d, fft_size);
end