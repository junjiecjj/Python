function CAP = OPT_CAP_MIMO(Heff,SNR);
S = svd(Heff)';
t = length(S);
CAP = 0;

while ~CAP
    onebylam = (SNR + sum(1./S(1:t).^2))/t;
    if onebylam - 1/S(t)^2 >= 0
        optP = onebylam - 1./S(1:t).^2;
        CAP = sum(log2(1+ S(1:t).^2 .* optP));
    else
        t = t-1;
    end
end