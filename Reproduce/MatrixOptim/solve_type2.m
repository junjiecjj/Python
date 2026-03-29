
function X_opt = solve_type2(H, S, P_T, N, L)
    % 求解问题：min ||H X - S||_F^2  s.t. (1/L) X X^H = (P_T / N) I_N
    % 等价于 X X^H = c I_N, c = L * P_T / N
    % 输入：H - M×N, S - M×L, P_T >0, N, L (要求 L ≥ N)
    % 输出：X_opt = sqrt(c) * V_A * U_{A,1}^H，其中 A = S^H H 的 SVD 分解
    [M, N_h] = size(H);
    if N_h ~= N
        error('H 的列数必须等于 X 的行数 N');
    end
    [M_s, L_s] = size(S);
    if M_s ~= M || L_s ~= L
        error('S 的尺寸必须与 H X 匹配');
    end
    if L < N
        error('约束要求 L ≥ N，否则无解');
    end
    if P_T <= 0
        error('P_T 必须为正数');
    end
    
    c = L * P_T / N;
    A = S' * H;                     % L×N
    [U_A, ~, V_A] = svd(A, 'econ'); % U_A: L×L, V_A: N×N (因为 L≥N)
    U_A1 = U_A(:, 1:N);             % L×N
    Z_opt = V_A * U_A1';            % N×L
    X_opt = sqrt(c) * Z_opt;
end