function [miCurve, gmiCurve, Hx] = ps64qam_air_curve(constellation, bitLabels, pmf, snrDbVec, Nsym)
% 计算MI与GMI曲线

Hx = ps64qam_entropy(pmf);
miCurve  = zeros(length(snrDbVec), 1);
gmiCurve = zeros(length(snrDbVec), 1);

for ii = 1:length(snrDbVec)
    [txIdx, txSym] = ps64qam_draw_symbols(constellation, pmf, Nsym);
    sigma2 = 10^(-snrDbVec(ii)/10);
    rxSym  = txSym + sqrt(sigma2/2) * (randn(Nsym,1) + 1i*randn(Nsym,1));

    [miCurve(ii), gmiCurve(ii)] = ps64qam_air_single(rxSym, txIdx, constellation, bitLabels, pmf, sigma2);
    fprintf('AIR进度: SNR = %+5.1f dB | MI = %.4f | GMI = %.4f\n', snrDbVec(ii), miCurve(ii), gmiCurve(ii));
end
end
