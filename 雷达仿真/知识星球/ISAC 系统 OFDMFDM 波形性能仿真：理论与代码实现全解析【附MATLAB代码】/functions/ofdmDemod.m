function Y = ofdmDemod(y_rx, K, cpLen)
% y_rx: (K+cpLen) x Nr (time)
y_noCP = y_rx(cpLen+1:end, :);            % K x Nr
Y = fft(y_noCP, K);                       % K x Nr
end
