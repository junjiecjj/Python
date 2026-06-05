% Tangent space projector
function PxZ = Px(X, Z, Pt)
    [N, M] = size(X);
    d = diag(Z*X');
    PxZ = Z - diag(d)*X*(N/(M*Pt));
end