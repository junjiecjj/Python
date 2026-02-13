function target = create_target_object(r, rvel, angle)
%CREATE_TARGET_OBJECT 创建一维目标对象结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含距离、速度和角度信息的一维目标对象结构体
% 版本: 1.0
% 输入参数:
%   r     - 目标距离 [m]
%   rvel  - 目标径向速度 [m/s]
%   angle - 目标角度 [deg]，默认0
% 输出参数:
%   target - 目标对象结构体，包含:
%            .r     - 距离 [m]
%            .rvel  - 径向速度 [m/s]
%            .angle - 角度 [deg]
%            .loc   - 3D位置 [x; y; z] [m]
%            .vel   - 3D速度 [vx; vy; vz] [m/s]

    if nargin < 3
        angle = 0;
    end
    
    target.r = r;
    target.rvel = rvel;
    target.angle = angle;
    
    ar = deg2rad(angle);
    
    target.loc = zeros(3, 1);
    target.loc(1) = r * sin(ar);
    target.loc(2) = r * cos(ar);
    target.loc(3) = 0;
    
    target.vel = zeros(3, 1);
    target.vel(1) = rvel * sin(ar);
    target.vel(2) = rvel * cos(ar);
    target.vel(3) = 0;
end
