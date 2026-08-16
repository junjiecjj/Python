






%% Local Functions
function ser = qpsk_ser_from_waveform(H, X, data, Q, N0)
    Kc = size(H, 1);
    L = size(X, 2);
    W = sqrt(N0 / 2) * (randn(Kc, L) + 1j * randn(Kc, L));
    Y = H * X + W;
    dataHat = pskdemod(Y, Q, pi / Q, 'gray');
    ser = mean(dataHat(:) ~= data(:));
end