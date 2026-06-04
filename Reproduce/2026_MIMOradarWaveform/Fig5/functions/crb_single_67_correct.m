
% 只对 n = (-(M-1)/2 : (M-1)/2).'; 适用
function CRB = crb_single_67_correct(a, adot, Rs, SNR)
    M = length(a);
    term1 = (a' * Rs * a) * (adot' * adot);
    term2 = M * (adot' * Rs * adot);
    term3 = M * abs(a' * Rs * adot)^2 / (a' * Rs * a);
    CRB = real(1 / (2 * SNR * (term1 + term2 - term3)));
end