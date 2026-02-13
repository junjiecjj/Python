function target = create_3d_target(r, rvel, angles)
%CREATE_3D_TARGET 创建三维目标对象结构体
% 作者: Radar Engineer
% 时间: 2024-07-23
% 功能: 创建包含距离、速度和俯仰/方位角信息的三维目标对象结构体
% 版本: 1.0
% 输入参数:
%   r      - 目标距离 [m]
%   rvel   - 目标径向速度 [m/s]
%   angles - 角度元组 [theta, phi] [deg]
%            theta: 俯仰角(仰角)，正值表示向上
%            phi:   方位角，正值表示顺时针方向
% 输出参数:
%   target - 目标对象结构体，包含:
%            .r     - 距离 [m]
%            .rvel  - 径向速度 [m/s]
%            .angle - 角度元组 [theta, phi] [deg]
%            .loc   - 3D位置 [x; y; z] [m]
%            .vel   - 3D速度 [vx; vy; vz] [m/s]

    target = create_target_object(0, 0, 0);
    
    theta = deg2rad(angles(1));
    phi = deg2rad(angles(2));
    
    target.r = r;
    target.rvel = rvel;
    target.angle = angles;
    
    target.loc = zeros(3, 1);
    target.loc(1) = r * cos(theta) * sin(phi);
    target.loc(2) = r * cos(theta) * cos(phi);
    target.loc(3) = r * sin(theta);
    
    target.vel = zeros(3, 1);
    target.vel(1) = rvel * cos(theta) * sin(phi);
    target.vel(2) = rvel * cos(theta) * cos(phi);
    target.vel(3) = rvel * sin(theta);
end
