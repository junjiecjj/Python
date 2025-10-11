function CAP = EQ_CAP_MIMO(Heff,SNR);

S = svd(Heff);
t = length(S);
CAP = sum(log2(1+ S.^2 * SNR/t));
    
