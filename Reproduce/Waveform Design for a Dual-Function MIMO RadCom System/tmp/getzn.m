
function zn = getzn(Z, M, P, n)
    zn = zeros(1, M);
    for p = 1:P
        zn = zn + Z(p:p+M-1, (n-1)*P + p).';   % 非共轭转置 -> 行向量
    end
    zn = zn / P;
end

