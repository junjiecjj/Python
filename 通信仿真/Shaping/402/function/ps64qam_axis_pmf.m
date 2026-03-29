function [pAxisUniform, pAxisShaped] = ps64qam_axis_pmf(axisLevels, constRaw, pmfUniform, pmfShaped)
% 统计I轴的一维边缘概率分布
pAxisUniform = zeros(length(axisLevels),1);
pAxisShaped  = zeros(length(axisLevels),1);

for k = 1:length(axisLevels)
    mask = real(constRaw) == axisLevels(k);
    pAxisUniform(k) = sum(pmfUniform(mask));
    pAxisShaped(k)  = sum(pmfShaped(mask));
end
end
