function a = steering_vector(phi, wavelength, d, element_num)
% 均匀线阵导向矢量
% phi            -- 目标方位 (0:180)/度
% wavelength     -- 波长
% d              -- 阵元间距
% element_num    -- 阵元数

n = (0 : element_num - 1).';
a = exp(j * 2 * pi * n * d * cosd(phi) ./ wavelength);