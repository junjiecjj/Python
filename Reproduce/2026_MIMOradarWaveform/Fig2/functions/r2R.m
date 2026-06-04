


% 定义从 r 到 R 的映射函数（假设 r 是按顺序排列的：对角元、上三角实部、上三角虚部）
function R = r2R(r, M)
    R = zeros(M, M);
    idx = 1;
    % 对角线
    for i = 1:M
        R(i,i) = r(idx);
        idx = idx + 1;
    end
    % 上三角
    for i = 1:M-1
        for j = i+1:M
            R(i,j) = r(idx) + 1j * r(idx+1);
            R(j,i) = conj(R(i,j));  % 利用共轭对称填充下三角
            idx = idx + 2;
        end
    end
end