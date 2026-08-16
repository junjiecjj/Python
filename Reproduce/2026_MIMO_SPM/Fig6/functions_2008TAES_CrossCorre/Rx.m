

% Retractor mapping
function RxZ = Rx(X, Z, Pt)
    [N, L] = size(X);
    Y = X + Z;
    d = diag(Y*Y').^(-1/2);
    RxZ = sqrt(L * Pt / N)*diag(d) * Y;
end