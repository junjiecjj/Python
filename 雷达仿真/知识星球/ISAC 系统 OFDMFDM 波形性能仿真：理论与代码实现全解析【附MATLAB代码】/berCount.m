function [err, total] = berCount(bits, bits_hat)
bits_hat = bits_hat(:);
L = min(length(bits), length(bits_hat));
err = sum(bits(1:L) ~= bits_hat(1:L));
total = L;
end
