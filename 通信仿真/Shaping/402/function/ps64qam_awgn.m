function rxSym = ps64qam_awgn(txSym, snrDb)
% 复高斯白噪声信道

sigma2 = 10^(-snrDb/10);
noise = sqrt(sigma2/2) * (randn(size(txSym)) + 1i*randn(size(txSym)));
rxSym = txSym + noise;
end
