% Retractor mapping
function RxZ = Rx(X, Z, Pt)
    [N, M] = size(X);
    Y = X + Z;
    d = diag(Y*Y').^(-1/2);
    RxZ = sqrt(M * Pt / N)*diag(d) * Y;
end