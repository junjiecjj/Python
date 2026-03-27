function x = golomb(N)
% x = golomb(N), generate a Golomb sequence x of length N

alpha = exp(1i * 2*pi /N);
exponent = (0:(N-1))' .* (1:N)' / 2;
x = alpha.^exponent;