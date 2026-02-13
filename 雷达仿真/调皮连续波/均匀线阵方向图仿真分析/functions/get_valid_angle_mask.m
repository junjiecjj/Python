function mask = get_valid_angle_mask(ura, angle_bins)
% GET_VALID_ANGLE_MASK 获取有效角度掩码
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算角度在[-π/2, π/2]范围内的有效掩码
% 版本: 1.0
% 输入参数:
%   ura - 均匀面阵结构体
%   angle_bins - 角度采样点
% 输出参数:
%   mask - 有效角度掩码(角度在[-π/2, π/2]范围内)

    mask = abs(angle_bins) <= (pi / 2);
end