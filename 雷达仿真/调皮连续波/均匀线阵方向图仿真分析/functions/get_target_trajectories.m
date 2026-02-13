function [r, v, a] = get_target_trajectories(target, t)
% GET_TARGET_TRAJECTORIES 获取目标在给定时间序列的轨迹
% 作者: Radar Engineer
% 时间: 2026-02-13
% 功能: 计算目标在给定时间序列的距离、速度和加速度
% 版本: 1.1
% 输入参数:
%   target - 目标对象结构体
%   t - 时间序列 (秒)
% 输出参数:
%   r - 距离轨迹 (米)
%   v - 速度轨迹 (米/秒)
%   a - 加速度轨迹 (米/秒^2)

    % 确保t是行向量
    if iscolumn(t)
        t = t';
    end
    
    % 计算距离轨迹（匀速运动）
    % 与Python保持一致: 距离保持不变，速度恒定
    r = repmat(target.r, size(t));
    
    % 计算速度轨迹
    v = repmat(target.rvel, size(t));
    
    % 计算加速度轨迹 (零加速度)
    a = zeros(size(t));
end