function A = steeringMatrixULA1D(normalizedPos, ang)
    % 输入：
    %   normalizedPos - 1×N 阵元位置，单位为 wavelength
    %   ang           - 方位角网格，单位 degree
    % 输出：
    %   A             - N×length(ang) 导向矢量矩阵
    normalizedPos = normalizedPos(:);
    A = exp(1j * 2 * pi * normalizedPos * sind(ang));
end