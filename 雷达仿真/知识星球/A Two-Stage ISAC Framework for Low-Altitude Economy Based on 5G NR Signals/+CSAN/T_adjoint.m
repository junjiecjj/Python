function Q = T_adjoint(P, M, N)
% T_adjoint
%
% 实现 T(Q) 的伴随算子 T*(P) (未归一化版本)。
% 这是我们推导出的 T_adj_sum。
%
% <P, T(Q)> = <T*(P), Q>
%
% T*(P) 的 (k, l) 元素 u_l(k) 是 P 中所有与 u_l(k) 相乘的元素的和。

    MN = M * N;
    Q = complex(zeros(2*M-1, 2*N-1));

    % 遍历 Q 的所有元素 (k, l)
    for l = -N+1 : N-1  % Q 的列索引
        
        Q_l = complex(zeros(M, M));
        
        % 1. 计算 Q_l = sum(P_{i,j}) for i-j=l
        for i = 1:N
            for j = 1:N
                if (i - j == l)
                    % 提取 P 的 (i,j) 块
                    P_ij = P((i-1)*M+1 : i*M, (j-1)*M+1 : j*M);
                    Q_l = Q_l + P_ij;
                end
            end
        end
        
        % 2. 计算 u_l(k) = trace(Q_l, k)
        u_l_vec = complex(zeros(2*M-1, 1));
        for k = -M+1 : M-1 % Q 的行索引
            
            % sum(diag(A, k)) 计算第 k 条对角线的和
            diag_sum = sum(diag(Q_l, -k));
            
            % 存储到 u_l 向量中 (MATLAB 索引: k+M)
            u_l_vec(k + M) = diag_sum;
        end
        
        % 将 u_l 向量存入 Q 矩阵
        Q(:, l + N) = u_l_vec;
    end
end
