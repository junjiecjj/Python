
function ser = qpsk_ser_zero_mui(S, data, Q, N0)
    [Kc, L] = size(S);
    W = sqrt(N0 / 2) * (randn(Kc, L) + 1j * randn(Kc, L));
    Y = S + W;
    dataHat = pskdemod(Y, Q, pi / Q, 'gray');
    ser = mean(dataHat(:) ~= data(:));
end