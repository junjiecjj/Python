
%% 辅助函数：构造 D 矩阵 (M^2 x L)
function D = construct_D(theta_vec, R_s, M, N)
    n = 0 : (M-1);
    a_func = @(th) exp(-1j * pi * n' * sind(th));
    A = @(th) a_func(th) * a_func(th).';
    [U, Lambda] = eig(R_s);
    U_sqrtL = U * sqrtm(Lambda);
    L = length(theta_vec);
    D = zeros(M^2, L);
    for l = 1:L
        D(:,l) = reshape( sqrt(N) * (A(theta_vec(l)) * U_sqrtL), [], 1);
    end
end