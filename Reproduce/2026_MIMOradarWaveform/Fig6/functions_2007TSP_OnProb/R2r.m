
% 从 R 得到 r 的逆函数（用于验证）
function r = R2r(R)
    M = size(R,1);
    r = zeros(M + M*(M-1), 1);
    idx = 1;
    for i = 1:M
        r(idx) = real(R(i,i));
        idx = idx + 1;
    end
    for i = 1:M-1
        for j = i+1:M
            r(idx) = real(R(i,j));
            r(idx+1) = imag(R(i,j));
            idx = idx + 2;
        end
    end
end