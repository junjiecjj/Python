function [miValue, gmiValue] = ps64qam_air_single(rxSym, txIdx, constellation, bitLabels, pmf, sigma2)
% 单个SNR点下的MI与GMI估计

N = length(rxSym);
M = length(constellation);
m = size(bitLabels, 2);

% 距离矩阵：N x M
distMat = abs(rxSym - transpose(constellation)).^2;

% log[ p(x) * p(y|x) ]
logMetric = -distMat ./ sigma2 + transpose(log(max(pmf(:), realmin)));

% log p(y)
logDen = ps64qam_logsumexp_rows(logMetric);

% ========================= MI =========================
distTx = abs(rxSym - constellation(txIdx)).^2;
logPyx = -distTx ./ sigma2;   
miValue = mean((logPyx - logDen) / log(2));

% ========================= GMI（按比特互信息求和） =========================
gmiValue = 0;
for k = 1:m
    mask0 = bitLabels(:,k) == 0;
    mask1 = ~mask0;

    pB0 = sum(pmf(mask0));
    pB1 = sum(pmf(mask1));

    logNum0 = ps64qam_logsumexp_rows(logMetric(:, mask0));
    logNum1 = ps64qam_logsumexp_rows(logMetric(:, mask1));

    txBits = bitLabels(txIdx, k);
    bitTerm = zeros(N,1);

    idx0 = (txBits == 0);
    idx1 = ~idx0;

    bitTerm(idx0) = (logNum0(idx0) - log(pB0) - logDen(idx0)) / log(2);
    bitTerm(idx1) = (logNum1(idx1) - log(pB1) - logDen(idx1)) / log(2);

    gmiValue = gmiValue + mean(bitTerm);
end
end
