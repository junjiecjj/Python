function mask = get_valid_angle_mask_mimo(mva, theta)
% GET_VALID_ANGLE_MASK_MIMO 获取MIMO虚拟阵列有效角度掩码
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 根据阵列维度获取相应的有效角度掩码
% 版本: 1.0
% 输入参数:
%   mva - MIMO虚拟阵列结构体
%   theta - 角度采样点
% 输出参数:
%   mask - 有效角度掩码

    if mva.array_dimension == 2
        mask = get_valid_angle_mask(mva.pa, theta);
    else
        mask = [];
    end
end