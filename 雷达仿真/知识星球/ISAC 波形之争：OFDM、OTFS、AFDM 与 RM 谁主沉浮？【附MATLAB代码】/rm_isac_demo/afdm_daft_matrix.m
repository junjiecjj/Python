function U = afdm_daft_matrix(L, c1, c2)
%AFDM_DAFT_MATRIX 教学型AFDM离散仿射傅里叶发送矩阵
%
% U = Lambda(c1) * F^H * Lambda(c2)
% 其中 Lambda(c) 为离散二次相位(chirp)对角矩阵。

n = (0:L-1).';
Fh = unitary_dft_matrix(L)';
Lambda1 = diag(exp(1j * pi * c1 * (n.^2)));
Lambda2 = diag(exp(1j * pi * c2 * (n.^2)));
U = Lambda1 * Fh * Lambda2;
end
