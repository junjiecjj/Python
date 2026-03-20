

function [J, r] = generateJr(M, R)
    % 计算 r 的长度
    n_diag = M;
    n_upper = M*(M-1)/2;
    len_r = n_diag + 2*n_upper; % 应该等于 M^2
    
    % 初始化 r 和 J
    r = zeros(len_r, 1);
    J = zeros(M^2, len_r); % 复数矩阵
    
    % 填充 r 并记录上三角元素的索引映射
    % 先填对角线
    for i = 1:M
        r(i) = real(R(i,i));
    end
    
    % 填上三角，同时记录每个 (i,j) 对应的实部和虚部在 r 中的位置
    % 使用两个矩阵 real_idx 和 imag_idx 来存储映射，方便后续填充 J
    real_idx = zeros(M, M);
    imag_idx = zeros(M, M);
    idx = M; % 当前 r 的索引，从对角线之后开始
    for i = 1:M-1
        for j = i+1:M
            idx = idx + 1;
            real_idx(i,j) = idx;
            r(idx) = real(R(i,j));
            idx = idx + 1;
            imag_idx(i,j) = idx;
            r(idx) = imag(R(i,j));
        end
    end
    
    % 现在填充 J
    % 遍历所有 (i,j) 位置，构造 vec(R) 的对应行
    for i = 1:M
        for j = 1:M
            row = (j-1)*M + i; % vec(R) 的索引（列堆叠）
            if i == j
                % 对角线元素
                col = i; % 对应 r 中的对角线位置
                J(row, col) = 1;
            elseif i < j
                % 上三角元素
                col_real = real_idx(i,j);
                col_imag = imag_idx(i,j);
                J(row, col_real) = 1;
                J(row, col_imag) = 1i;
            else % i > j
                % 下三角元素，利用共轭对称性：R(i,j) = conj(R(j,i))
                col_real = real_idx(j,i);
                col_imag = imag_idx(j,i);
                J(row, col_real) = 1;
                J(row, col_imag) = -1i;
            end
        end
    end
end