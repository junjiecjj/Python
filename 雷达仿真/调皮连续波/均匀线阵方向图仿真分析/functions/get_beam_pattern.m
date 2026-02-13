function [p_dB, angle_bins] = get_beam_pattern(ula, theta0, n_theta)
% GET_BEAM_PATTERN 计算波束方向图(解析解)
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 通过均匀线阵的经典方向图公式计算指定波束指向角下的阵列方向图
% 版本: 1.0
% 输入参数:
%   ula     - 均匀线阵结构体
%   theta0  - 波束指向角 [rad]
%   n_theta - 角度采样数，默认256
% 输出参数:
%   p_dB       - 波束方向图 [dB]，已归一化
%   angle_bins - 角度采样点 [rad]

    if nargin < 3
        n_theta = 256;
    end
    
    angle_bins = get_angle_bins_full(ula, n_theta);
    pdss = pi * ula.d * (sin(angle_bins) - sin(theta0));
    
    p = ones(size(pdss));
    idx = pdss ~= 0;
    p(idx) = sin(ula.nr * pdss(idx)) ./ (ula.nr * sin(pdss(idx)));
    
    p_dB = 20 * log10(abs(p));
    p_dB = p_dB - max(p_dB);
end