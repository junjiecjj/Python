function probGrid = ps64qam_probability_grid(constellation, pmf, axisLevels)
% 根据真实星座坐标把64个概率映射到8x8二维网格
% 行对应Q轴从小到大，列对应I轴从小到大

numLevels = length(axisLevels);
probGrid = zeros(numLevels, numLevels);

constellation = constellation(:);
pmf = pmf(:);

for k = 1:length(constellation)
    iVal = real(constellation(k));
    qVal = imag(constellation(k));

    iIdx = find(axisLevels == iVal, 1, 'first');
    qIdx = find(axisLevels == qVal, 1, 'first');

    if isempty(iIdx) || isempty(qIdx)
        error('ps64qam_probability_grid: 星座坐标与axisLevels不匹配。');
    end

    probGrid(qIdx, iIdx) = probGrid(qIdx, iIdx) + pmf(k);
end
end
