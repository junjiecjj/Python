function s_t = ofdmMod(X, K, cpLen)
% X: Kx1 (freq)
x_t = ifft(X, K);                         % Kx1
cp = x_t(end-cpLen+1:end);
s_t = [cp; x_t];                          % (K+cpLen)x1
end
