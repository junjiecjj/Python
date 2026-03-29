function detIdx = ps64qam_map_detect(rxSym, constellation, pmf, sigma2)
% MAP检测器
distMat = abs(rxSym - transpose(constellation)).^2;
metric  = -distMat ./ sigma2 + transpose(log(max(pmf(:), realmin)));
[~, detIdx] = max(metric, [], 2);
end
