


function U = initRowSemiUnitary(N, L)
    Z = randn(L, N) + 1j * randn(L, N);
    [Q, ~] = qr(Z, 0);
    U = Q';
end