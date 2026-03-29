function berCurve = ps64qam_ber_curve(constellation, bitLabels, pmf, snrDbVec, Nsym)
% BER曲线计算
% 判决方式采用MAP形式：
% argmax_x [ log p(x) + log p(y|x) ]
% 对于均匀分布，这个MAP就退化为普通ML判决

berCurve = zeros(length(snrDbVec), 1);

for ii = 1:length(snrDbVec)
    [txIdx, txSym] = ps64qam_draw_symbols(constellation, pmf, Nsym);
    sigma2 = 10^(-snrDbVec(ii)/10);
    rxSym  = txSym + sqrt(sigma2/2) * (randn(Nsym,1) + 1i*randn(Nsym,1));

    detIdx = ps64qam_map_detect(rxSym, constellation, pmf, sigma2);

    txBits = bitLabels(txIdx, :);
    rxBits = bitLabels(detIdx, :);

    berCurve(ii) = mean(txBits(:) ~= rxBits(:));
    fprintf('BER进度: SNR = %+5.1f dB | BER = %.6e\n', snrDbVec(ii), berCurve(ii));
end
end
