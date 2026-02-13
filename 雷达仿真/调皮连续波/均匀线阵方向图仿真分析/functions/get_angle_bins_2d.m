function [theta_bins, phi_bins] = get_angle_bins_2d(ura, d, fft_size)
% GET_ANGLE_BINS_2D 获取2D角度采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算基于FFT的2D角度采样点
% 版本: 1.0
% 输入参数:
%   ura - 均匀面阵结构体
%   d        - 归一化间距
%   fft_size - FFT点数 [N_fft_el × N_fft_az]
% 输出参数:
%   theta_bins - 俯仰角采样点 [rad], 一维向量
%   phi_bins   - 方位角采样点 [rad], 一维向量

    if isscalar(fft_size)
        fft_size = [fft_size, fft_size];
    end
    
    % 俯仰角采样 (对应z轴/行方向)
    radrange_el = -pi : 2*pi/fft_size(1) : pi-2*pi/fft_size(1);
    theta_bins = asin(radrange_el / (2 * pi * d));
    
    % 方位角采样 (对应x轴/列方向)
    % 使用中间俯仰角(theta≈0)时的方位角范围
    radrange_az = -pi : 2*pi/fft_size(2) : pi-2*pi/fft_size(2);
    phi_bins = asin(radrange_az / (2 * pi * d));
end