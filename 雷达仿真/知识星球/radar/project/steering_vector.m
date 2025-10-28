function sv = steering_vector(pos, ang, lambda)
    %steering_vector 计算导向矢量
    %   ANG 表示输入信号的方向。ANG 可以是 1xM 向量或 2xM 矩阵，其中 M 是输入信号的数量。
    %   如果 ANG 是一个 2xM 矩阵，每列以 [方位角; 俯仰角] 的形式（以度为单位）指定空间中的方向。
    %   方位角必须在 -180° 到 180° 之间，俯仰角必须在 -90° 到 90° 之间。
    %   方位角定义在 xy 平面内；它是与 x 轴（也是阵列法线方向）的夹角， y 轴正方向对应 90°。
    %   俯仰角定义为与 xy 平面的夹角
    %
    %   % Example:
    %   lambda = 1.25e-2;
    %   pos = [[0; 1; 0], [0; 5; 0], [0; 2; 0], [0; 3; 0]] * 0.5 * lambda;
    %   ang = [[10; 0], [20; 10], [30; 20], [0; 30]];
    %   my_sv = steering_vector(pos, ang, lambda);

    % 方位角
    azimuth = ang(1, :);
    % 俯仰角
    elevation = ang(2, :);
    % 目标方向的 单位方向向量
    unit_direction_vector = [cosd(elevation) .* cosd(azimuth); cosd(elevation) .* sind(azimuth); sind(elevation)];
    % 信号到达不同阵元的距离差，参考阵元坐标(0;0;0)
    distance = transpose(pos) * unit_direction_vector;
    % 计算导向矢量
    sv = exp(1j * 2 * pi * distance / lambda);
end
