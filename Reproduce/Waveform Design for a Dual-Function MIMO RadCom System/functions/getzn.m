


function zn = getzn(Z, M, P, n)
    zn = zeros(M, P);
    for i = 1:P
        zn(:, i) = Z(i:i+M-1, (n-1)*P + i);
    end

    zn = mean(zn, 2);
end