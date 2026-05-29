function R = recover_R_from_r(r, M)
    % 从参数向量 r 恢复 Hermitian 矩阵 R
    % 输入: r - 长度为 M^2 的实向量（按对角线+上三角实部+上三角虚部顺序）
    %       M - 矩阵维度
    % 输出: R - M×M Hermitian 矩阵

    if length(r) ~= M^2
        error('r 的长度必须为 M^2');
    end

    % 初始化
    R = zeros(M, M);

    % 对角线
    for i = 1:M
        R(i,i) = r(i);
    end

    % 填充上三角
    idx = M; % 当前已用索引
    for i = 1:M-1
        for j = i+1:M
            real_part = r(idx+1);
            imag_part = r(idx+2);
            R(i,j) = real_part + 1j * imag_part;
            idx = idx + 2;
        end
    end

    % 利用共轭对称填充下三角
    for i = 1:M-1
        for j = i+1:M
            R(j,i) = conj(R(i,j));
        end
    end
end