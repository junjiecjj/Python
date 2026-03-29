function bits = ps64qam_int2bits(intVal, nBits)
% 把十进制整数转成固定长度二进制比特行向量（左高位）
bits = zeros(1, nBits);
for k = 1:nBits
    bits(nBits-k+1) = bitget(intVal, k);
end
end
