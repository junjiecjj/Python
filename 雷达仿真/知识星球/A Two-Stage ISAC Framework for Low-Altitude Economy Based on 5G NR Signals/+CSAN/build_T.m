function T_U = build_T(U, M, N)
% build_T
%
% 实现 T(U) 算子 (公式 23)。
% 从 (2M-1)x(2N-1) 的 U 构建 MN x MN 的块托普利茨矩阵。

    MN = M * N;
    T_U = complex(zeros(MN, MN));
    
    % 遍历 T_U 的所有 M x M 块
    for i = 1:N % 块行索引
        for j = 1:N % 块列索引
            
            % 块 (i,j) 对应的 u_l 的索引是 l = i - j
            l = i - j;
            
            % 从 U 中获取对应的 u_l 向量
            % (U 的列索引 l 映射到 MATLAB 的 l + N)
            u_l = U(:, l + N);
            
            % 构建 Toep(u_l)
            % 第 1 列: u_l(0), u_l(1), ..., u_l(M-1)
            % (MATLAB 索引: k=0 映射到 M)
            col = u_l(M : 2*M-1);
            
            % 第 1 行: u_l(0), u_l(-1), ..., u_l(-M+1)
            % (MATLAB 索引: k=0 映射到 M)
            row = u_l(M : -1 : 1);
            
            Toep_l = toeplitz(col, row);
            
            % 将块放入 T_U
            T_U((i-1)*M+1 : i*M, (j-1)*M+1 : j*M) = Toep_l;
        end
    end
end
