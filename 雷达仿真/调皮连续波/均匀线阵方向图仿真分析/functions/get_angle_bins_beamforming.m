function angle_bins = get_angle_bins_beamforming(bf)
% GET_ANGLE_BINS_BEAMFORMING 获取波束形成角度采样点
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 获取波束形成DOA估计的角度采样点
% 版本: 1.0
% 输入参数:
%   bf - 波束形成DOA估计结构体
% 输出参数:
%   angle_bins - 角度采样点

    angle_bins = bf.angle_bins;
end