function bitsHat = qpsk_hard_demod(sym)
%QPSK_HARD_DEMOD QPSK硬判决解调
sym = sym(:);
bitsHat = zeros(2 * numel(sym), 1);
bitsHat(1:2:end) = real(sym) >= 0;
bitsHat(2:2:end) = imag(sym) >= 0;
end
