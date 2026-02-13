function sv = get_steering_vector(ula, theta)
% GET_STEERING_VECTOR 获取导向向量
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 计算均匀线阵在给定角度的导向向量
% 版本: 1.0
% 输入参数:
%   ula - 均匀线阵结构体
%   theta - 目标角度 [rad]
% 输出参数:
%   sv - 导向向量 [N × 1]

    n = 0:ula.nr-1;
    sv = exp(2i * pi * ula.d * n * sin(theta));
    sv = sv(:);
end