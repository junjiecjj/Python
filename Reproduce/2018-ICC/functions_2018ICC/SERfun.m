

function ser = SERfun(H, X, data, Q)
    rd = pskdemod(H*X, Q, pi/Q);
    [~, ser] = symerr(data, rd);
end