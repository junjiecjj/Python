function zn = getzn(Z, M, P, n)
% 从扩展矩阵 Z 中提取第 n 个波形的估计（列向量）
% Z: (P+M-1) × (N*P)
    zn = zeros(M, 1);
    for p = 1:P
        zn = zn + Z(p:p+M-1, (n-1)*P + p);
    end
    zn = zn / P;
end