function sym = qpsk_mod(bits)
%QPSK_MOD 硬映射QPSK，Gray风格符号顺序
bits = bits(:);
assert(mod(numel(bits), 2) == 0, 'Number of bits must be even.');

bI = 2 * bits(1:2:end) - 1;
bQ = 2 * bits(2:2:end) - 1;
sym = (bI + 1j * bQ) / sqrt(2);
end
