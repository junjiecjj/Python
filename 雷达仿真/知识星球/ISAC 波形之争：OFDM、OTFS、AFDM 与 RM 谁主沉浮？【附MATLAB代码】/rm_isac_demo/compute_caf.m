function CAF = compute_caf(rxSig, refSig, delayGrid, dopplerGrid)
%COMPUTE_CAF 交叉模糊函数(CAF)
%
% CAF(k, l) = sum_n r[n] * conj(x[n-l]) * exp(-j2*pi*nu*n/L)

L = numel(refSig);
n = (0:L-1).';
CAF = zeros(numel(dopplerGrid), numel(delayGrid));

for idop = 1:numel(dopplerGrid)
    nu = dopplerGrid(idop);
    dopplerComp = exp(-1j * 2*pi * nu * n / L);

    for idel = 1:numel(delayGrid)
        tau = delayGrid(idel);
        refShift = circshift(refSig, tau);
        CAF(idop, idel) = sum(rxSig .* conj(refShift) .* dopplerComp);
    end
end

end
