function traj = get_target_trajectories_3d(target, t)
% GET_TARGET_TRAJECTORIES_3D 获取3D目标在给定时间序列的轨迹
% 作者: Radar Engineer
% 时间: 2026-02-13
% 功能: 计算3D目标在给定时间序列的位置轨迹
% 版本: 1.2
% 输入参数:
%   target - 3D目标对象结构体
%   t - 时间序列或单个时间点 (秒)
% 输出参数:
%   traj - 3D轨迹
%          - 单个时间点: [3 × 1]
%          - 时间序列: [3 × N_timesteps]

    % 处理单个时间点的情况
    if isscalar(t)
        % 计算3D位置（匀速运动）
        traj = zeros(3, 1);
        traj(1) = target.loc(1) + target.vel(1) * t;
        traj(2) = target.loc(2) + target.vel(2) * t;
        traj(3) = target.loc(3) + target.vel(3) * t;
    else
        % 确保t是列向量
        if isrow(t)
            t = t';
        end
        
        % 计算3D位置轨迹（匀速运动）
        % 与Python保持一致: ret = self.loc + self.vel * (ts[None, :])
        traj = zeros(3, length(t));
        traj(1, :) = target.loc(1) + target.vel(1) * t';
        traj(2, :) = target.loc(2) + target.vel(2) * t';
        traj(3, :) = target.loc(3) + target.vel(3) * t';
    end
end