function F = unitary_dft_matrix(L)
%UNITARY_DFT_MATRIX 生成归一化DFT矩阵
n = (0:L-1).';
k = 0:L-1;
F = exp(-1j * 2*pi/L * (n * k)) / sqrt(L);
end
